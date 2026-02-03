import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def create_dp_sharding() -> tuple[NamedSharding, NamedSharding]:
    """
    Data Parallel
    """
    devices = jax.devices()
    mesh = Mesh(mesh_utils.create_device_mesh((len(devices),)), axis_names=("data",))
    data_sharding = NamedSharding(mesh, P("data", None))  # Batch
    replicated = NamedSharding(mesh, P())  # replicated params/scalars

    return data_sharding, replicated
