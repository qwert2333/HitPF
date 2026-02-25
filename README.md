# HitPF

**HitPF** is a GATr-based particle-flow reconstruction model for the CLD detector at the FCC-ee.
It performs two sequential tasks:

1. **Clustering** — groups calorimeter hits and tracks into particle-flow objects using an object-condensation loss.
2. **Property regression** — regresses a correction factor for each reconstructed cluster using a GNN-based model and a PID class

---

## Dependencies

The code can be used with this container:
```docker://dologarcia/gatr:v9```

---

## Dataset 

Input data is stored as `.parquet` files, each file stores 100 events. A sample of the dataset in ML-ready format can be found at [1](https://zenodo.org/records/18749298). The full dataset is hosted on CERN's EOS space.


---

## Training

### Step 1 — Clustering

```bash
bash scripts/train_clustering.sh
```


### Step 2 — Energy correction


```bash
bash scripts/train_energy_pid.sh
```

### Validation 

```bash
bash scripts/evaluation.sh
```

---

## Citation

If you use this code, please cite:

```bibtex
@software{hitpf2026,
  title  = {End-to-end event reconstruction for precision physics at future colliders code},
  year   = {2026},
  url    = {https://github.com/mgarciam/HitPF}
}
```
