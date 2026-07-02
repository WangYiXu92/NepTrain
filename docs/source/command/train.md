# train
**Description:**  
Performs automatic training for NEP.
## Input Parameters

:::{important}
- You should use `NepTrain init <type>` to generate `job.yaml`, and then use `NepTrain train job.yaml` to start the training task.

- After the program runs, a `restart.yaml` file will be generated. To continue training, you can use `NepTrain train restart.yaml`.
::: 
**Usage:**  
```bash
NepTrain train <config_path>
```

**Options:**
- `<config_path>`
  Path to configuration file, such as `job.yaml`.
## Output
During training, intermediate data and the resulting potential are
written to the working directory. A `restart.yaml` file is created after
each iteration, allowing you to resume the workflow with
`NepTrain train restart.yaml`.

NepTrain also writes non-fatal workflow audit reports to:

```text
<work_path>/workflow_reports/Generation-<N>/<stage>.json
```

Each report records the stage (`nep`, `gpumd`, `select`, `dft`, or `pred`),
status, artifact paths, file sizes, extxyz frame counts, and missing required
arrays/results when relevant. `NepTrain status <work_path>` summarizes these
reports together with the current generation and latest NEP loss, so interrupted
runs can be audited without guessing from file existence alone.

## Example

### Initialization
Run `NepTrain init slurm` on the command line to generate `job.yaml`, which contains all modifiable control parameters of the workflow.  
The command produces a `job.yaml` file showing default execution parameters and their explanations. Set the parameters line by line on the first run, and later reuse the file as a template for future training.
:::{tip}
If you copy a previously modified `job.yaml`, you can place it in the working directory and run `NepTrain init slurm` again to synchronize any newly added parameters during version updates.
:::

### Start Training 
Run the following command on the login node:
```shell
NepTrain train job.yaml
```
To run in the background:
```shell
nohup NepTrain train job.yaml &
```
