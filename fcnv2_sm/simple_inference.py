import datetime
import os

import numpy as np

# Set number of GPUs to use to 1
os.environ["WORLD_SIZE"] = "1"
# Set model registry as a local folder
model_registry = "/scratch/gilbreth/gupt1075/earth2mip"

# os.path.join(os.path.dirname(os.path.realpath(os.getcwd())), "models")


os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import inference_ensemble, registry
from earth2mip.initial_conditions import cds
from earth2mip.networks.fcnv2_sm import load as fcnv2_sm_load

# %%
# Load model(s) from registry
package = registry.get_model("fcnv2_sm")
print("loading FCNv2 small model, this can take a bit")
sfno_inference_model = fcnv2_sm_load(package)

data_source = cds.DataSource(sfno_inference_model.in_channel_names)
output = "/scratch/gilbreth/gupt1075/earth2mip/fcnv2_sm/output/"
time = datetime.datetime(2018, 1, 1)
ds = inference_ensemble.run_basic_inference(
    sfno_inference_model,
    n=1,
    data_source=data_source,
    time=time,
)


# %%
from scipy.signal import periodogram

arr = ds.sel(channel="u200").values
f, pw = periodogram(arr, axis=-1, fs=1)
pw = pw.mean(axis=(1, 2))
import matplotlib.pyplot as plt

l = ds.time - ds.time[0]  # noqa
days = l / (ds.time[-1] - ds.time[0])
cm = plt.cm.get_cmap("viridis")
for k in range(ds.sizes["time"]):
    day = (ds.time[k] - ds.time[0]) / np.timedelta64(1, "D")
    day = day.item()
    plt.loglog(f, pw[k], color=cm(days[k]), label=day)
plt.legend()
plt.ylim(bottom=1e-8)
plt.grid()
plt.savefig("u200_spectra.png")

# %%
day = (ds.time - ds.time[0]) / np.timedelta64(1, "D")
plt.semilogy(day, pw[:, 100:].mean(-1), "o-")
plt.savefig("/scratch/gilbreth/gupt1075/earth2mip/fcnv2_sm/output/u200_high_wave.png")
