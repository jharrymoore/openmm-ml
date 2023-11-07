
import torch


from openmmtools.testsystems import AlanineDipeptideVacuum

# from torch_nl import compute_neighborlist
from e3nn.util import jit
from mace.tools import utils, to_one_hot, atomic_numbers_to_indices

from typing import Iterable, List, Optional

from ase.units import kJ, mol, nm
from openmmml.models.utils import simple_nl, nnpops_nl
from openmm.unit import angstrom
from NNPOps.neighbors import getNeighborPairs

class MACEForce(torch.nn.Module):
    def __init__(
        self,
        model_path,
        atomic_numbers,
        indices,
        periodic,
        device,
        nl = "nnpops",
        dtype=torch.float64,
    ):
        super(MACEForce, self).__init__()
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )

        else:  # unless user has specified the device
            self.device = torch.device(device)

        # specify the neighbour list
        # if nl == "torch":
        #     self.nl = simple_nl
        # elif nl == "nnpops":
        
        # else:
            # raise ValueError(f"Neighbour list {nl} not recognised")

        self.default_dtype = dtype
        torch.set_default_dtype(self.default_dtype)

        print(
            "Running MACEForce on device: ",
            self.device,
            " with dtype: ",
            self.default_dtype,
            "and neigbbour list: ",
            nl,
        )
        self.periodic = periodic
        # conversion constants
        self.nm_to_distance = 10.0  # nm->A
        self.distance_to_nm = 0.1  # A->nm
        self.energy_to_kJ = mol / kJ  # eV->kJ


        self.model = torch.load(model_path, map_location=device)
        self.model.to(self.default_dtype)
        self.model.eval()

        # print(self.model)
        # for name, param in self.model.state_dict().items():
        #    print(name, param.size())

        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )

        self.model = jit.compile(self.model)

        # setup input
        N = len(atomic_numbers)
        self.ptr = torch.tensor([0, N], dtype=torch.long, device=self.device)
        self.batch = torch.zeros(N, dtype=torch.long, device=self.device)

        # one hot encoding of atomic number
        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(atomic_numbers, z_table=self.z_table),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(-1),
            num_classes=len(self.z_table),
        )

        if periodic:
            self.pbc = torch.tensor([True, True, True], device=self.device)
        else:
            self.pbc = torch.tensor([False, False, False], device=self.device)

        if indices is None:
            self.indices = None
        else:
            self.indices = torch.tensor(indices, dtype=torch.int64)

    def forward(self, positions, boxvectors: Optional[torch.Tensor] = None):
        # setup positions

        # positions = positions.to(device=self.device, dtype=self.default_dtype)
        # if self.indices is not None:
        #     positions = positions[self.indices]

        positions = positions * self.nm_to_distance

        if boxvectors is not None:
            cell = (
                boxvectors.to(device=self.device, dtype=self.default_dtype)
                * self.nm_to_distance
            )
        else:
            cell = torch.eye(3, device=self.device)

        # mapping, shifts_idx = nnpops_nl(
        #     positions, cell, self.periodic, self.r_max
        # )
        mapping, shifts_idx, distances, n_pairs = getNeighborPairs(positions, self.r_max)

        mapping = mapping.to(torch.int64)

        edge_index = torch.stack((mapping[0], mapping[1]))

        shifts = torch.mm(shifts_idx, cell)

        # create input dict
        input_dict = {
            "ptr": self.ptr,
            "node_attrs": self.node_attrs,
            "batch": self.batch,
            "pbc": self.pbc,
            "cell": cell,
            "positions": positions,
            "edge_index": edge_index,
            "unit_shifts": shifts_idx,
            "shifts": shifts,
        }

        # predict
        out = self.model(
            input_dict,
            compute_force=False,
            # particle_filter_indices=self.particle_filter_indices,
        )

        energy = out["interaction_energy"]
        if energy is None:
            energy = torch.tensor(0.0, device=self.device)

        # return energy
        energy = energy * self.energy_to_kJ

        return energy

        # Create the PyTorch model that will be invoked by OpenMM.

        # if extra particles specified, add these to the included atoms



test_system = AlanineDipeptideVacuum()
topology = test_system.topology
system = test_system.system

positions = torch.tensor(test_system.positions.value_in_unit(angstrom)).to(torch.device("cuda")).to(torch.float64)

print(positions.shape)



atomic_numbers = [atom.element.atomic_number for atom in topology.atoms()]

# torch_dtype = {"float32":torch.float32, "float64":torch.float64}[dtype]
maceforce = MACEForce(
    "/projects/mai/users/kmzp800_harry/mace-bio-simulations/SPICE_sm_inv_neut_E0_swa.model",
    atomic_numbers,
    [i for i in range(len(atomic_numbers))],
    False,
    "cuda",
    dtype=torch.float64,
)

# Convert it to TorchScript and save it.
module = torch.jit.script(maceforce)



s = torch.cuda.Stream()
with torch.cuda.stream(s):
    for i in range(10):
        output = module(positions)
torch.cuda.current_stream().wait_stream(s)


g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    output = module(positions)

for _ in range(100):
    g.replay()

    

    # attempt to do the graph compile thing

    # need to write these to the current working directory,
    #  such that they're not lost in a scratch directory
    #  and the simulation can be resulted
    # os.makedirs(os.path.join(os.getcwd(), "compiled_models"), exist_ok=True)
    # _, filename = tempfile.mkstemp(
    #     suffix=".pt", dir=os.path.join(os.getcwd(), "compiled_models")
    # )
    # module.save(filename)

    # # Create the TorchForce and add it to the System.
    # force = openmmtorch.TorchForce(filename)
    # force.setForceGroup(forceGroup)
    # force.setUsesPeriodicBoundaryConditions(is_periodic)
    # force.setProperty("useCUDAGraphs", "true")
    # # force.setOutputsForces(True)
    # system.addForce(force)