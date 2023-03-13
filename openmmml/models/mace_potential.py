from e3nn.util import jit
import torch
from torch_nl import compute_neighborlist, compute_neighborlist_n2
from mace import data
from mace.tools import torch_geometric, utils
from typing import Optional, Iterable
from ase import Atoms
import openmm
from openmmtorch import TorchForce
from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory
from ase import Atoms
from openmm.app import Topology

from ase.units import kJ, mol, nm


def compile_model(model_path):
    model = torch.load(model_path)
    res = {}
    res["model"] = jit.compile(model)
    res["z_table"] = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    print("rmax from model is", model.r_max)
    res["r_max"] = model.r_max
    return res


class MacePotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name: str, **args) -> MLPotentialImpl:
        return MacePotentialImpl(name)


class MacePotentialImpl(MLPotentialImpl):
    """
    Implements the mace potential in openMM, using TorchForce to add it to an openMM system
    """

    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        filename: str,
        implementation: str = "nnpops",
        **kwargs,
    ):
        class MACE_openmm(torch.nn.Module):
            def __init__(
                self,
                model_path: str,
                dtype: torch.dtype,
                pbc: bool,
                atom_indices: Optional[Iterable] = None,
                nl: str = "torch_nl",
                atoms_obj: Optional[Atoms] = None,
                device: str = "cuda",
            ):
                super().__init__()
                if nl == "torch_nl":
                    self.nl = compute_neighborlist
                elif nl == "torch_nl_n2":
                    self.nl = compute_neighborlist_n2
                else:
                    raise NotImplementedError
                self.device = torch.device(device)
                self.atom_indices = torch.tensor(atom_indices) if atom_indices is not None else None
                self.dtype = dtype
                self.pbc = pbc
                self.register_buffer("ev_to_kj_mol", torch.tensor(mol / kJ))
                self.register_buffer("eV_per_A_to_kj_mol_nm", torch.tensor((mol * nm) / kJ ))
                dat = compile_model(model_path)
                # TODO: if only the topology was passed, create the config from this
                # if multiple atoms objects, process them in a batch
                
                configs = data.config_from_atoms(atoms_obj)
                data_loader = torch_geometric.dataloader.DataLoader(
                    dataset=[
                        data.AtomicData.from_config(
                            configs, z_table=dat["z_table"], cutoff=dat["r_max"]
                        )
                    ],
                    batch_size=1,
                    shuffle=False,
                    drop_last=False,
                )
                batch = next(iter(data_loader)).to(self.device)
                batch_dict = batch.to_dict()
                batch_dict.pop("edge_index")
                batch_dict.pop("energy", None)
                batch_dict.pop("forces", None)
                batch_dict.pop("positions")
                # batch_dict.pop("shifts")
                batch_dict.pop("weight")
                self.inp_dict = batch_dict
                self.model = dat["model"]
                self.r_max = dat["r_max"]

            def forward(self, positions: torch.Tensor, boxVectors: Optional[torch.Tensor] = None):
                positions = (
                    positions[self.atom_indices] if self.atom_indices is not None else positions
                )
                # preallocate output energies
                if boxVectors is not None:
                    boxVectors = boxVectors * 10
                    boxVectors = boxVectors.type(self.dtype).to(self.device)
                else:
                    boxVectors = torch.zeros(3,3).type(self.dtype).to(self.device)
               
               
                positions = positions * 10
                
                bbatch = torch.zeros(positions.shape[0], dtype=torch.long, device=self.device)
                # slice positions using number of atoms in the config
            # else:
                mapping, batch_mapping, shifts_idx = self.nl(
                    cutoff=self.r_max,
                    pos=positions.to(self.device),
                    cell=boxVectors,
                    pbc=torch.tensor(3 * [self.pbc], device=self.device, requires_grad=False),
                    batch=bbatch,
                    dtype=self.dtype,
                )
                    # Eliminate self-edges that don't cross periodic boundaries
                true_self_edge = mapping[0] == mapping[1]
                true_self_edge &= torch.all(shifts_idx == 0, dim=1)
                keep_edge = ~true_self_edge

                # Note: after eliminating self-edges, it can be that no edges remain in this system
                sender = mapping[0][keep_edge]
                receiver = mapping[1][keep_edge]
                shifts_idx = shifts_idx[keep_edge]

                edge_index = torch.stack((sender, receiver))

                # From the docs: With the shift vector S, the distances D between atoms can be computed from
                inp_dict_this_config = self.inp_dict.copy()
                inp_dict_this_config["positions"] = positions.to(self.device)
                inp_dict_this_config["edge_index"] = edge_index
                inp_dict_this_config["shifts"] = shifts_idx
                res = self.model(inp_dict_this_config, compute_force=False)
                interaction_energy = res["interaction_energy"]
                if interaction_energy is None:
                    print("Interaction energy is None")
                    interaction_energy = torch.tensor(0.0, device=self.device)

                openmm_forces = res["forces"]
                if openmm_forces is None:
                    openmm_forces = torch.zeros(len(positions), 3, device=self.device)

                openmm_forces = openmm_forces * self.eV_per_A_to_kj_mol_nm 
                # assign to output energies

                return interaction_energy* self.ev_to_kj_mol
            
        try:
            model = torch.load(filename)
        except RuntimeError:
            model = torch.load(filename, map_location=torch.device("cpu"))
        model = jit.compile(model)
        print("MACE model compiled")
        # TODO: this should take a topology
        # A bit hacky to add the atoms object like this
        is_periodic = (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions()
        openmm_calc = MACE_openmm(filename, atom_indices=atoms, pbc=is_periodic, **kwargs)
        module = jit.script(openmm_calc)
        module.save("md_test_mace.pt")
        force = TorchForce("md_test_mace.pt")
        force.setOutputsForces(False)
        force.setForceGroup(forceGroup)
        
        # print("Periodic boundary conditions:", is_periodic)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        # force.setForceGroup()
        # modify the system in place to add the force
        system.addForce(force)
