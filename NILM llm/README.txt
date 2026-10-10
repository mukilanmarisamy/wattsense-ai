=====================================================================
WATTSENSE AI - Household Appliance Power Disaggregation (NILM)
=====================================================================

Households can measure their TOTAL electricity use, but not which
appliance caused it. This project demonstrates Non-Intrusive Load
Monitoring (NILM): from a single aggregate power reading, a neural
network (Seq2Point) estimates how much power an appliance used, when
it ran, and how many kWh (and how much money) it consumed. Two
interactive dashboards (built with Gradio) let you generate a
simulated household day and see the result.

NOTE: Everything runs on SIMULATED data. See section 5 (Current scope
and limitations) for exactly what the model learns and what is
simulated.


---------------------------------------------------------------------
 REQUIREMENTS
---------------------------------------------------------------------

Python libraries:

  tensorflow    Building, training and running the Seq2Point model
  numpy         Numbers and sliding windows
  pandas        Data tables and CSV files
  matplotlib    All plots
  gradio        The interactive dashboards
  scikit-learn  Installed by the notebook; not needed by the final cells

The standard-library modules random, datetime, os and sys need no
installation.

Install everything:

    pip install tensorflow pandas numpy matplotlib scikit-learn gradio

Python version: TensorFlow often supports Python versions a little
behind the newest release. If "pip install tensorflow" fails on your
machine, either use Google Colab (recommended, nothing to install) or
create a virtual environment with an older Python (for example 3.11
or 3.12) and check TensorFlow's install page for supported versions.


---------------------------------------------------------------------
 HOW TO USE THE NOTEBOOK
---------------------------------------------------------------------

OPTION A - Google Colab (recommended)

  1. Go to https://colab.research.google.com and choose
     File > Upload notebook, then select Untitled1.ipynb.
  2. Run the cells from top to bottom with Runtime > Run all, or press
     Shift+Enter cell by cell.
  3. The first cells install packages, so allow a few minutes. Colab
     may ask you to restart the runtime after installing. If so,
     restart and continue from the next cell.
  4. Training takes a few minutes (5 epochs). Wait for the message
     "Training complete!" before running the evaluation and dashboard
     cells.
  5. At the end, each dashboard cell shows a Gradio panel inside the
     notebook. Click the red button to run a simulation.

OPTION B - Jupyter on your own computer

    pip install jupyter tensorflow pandas numpy matplotlib scikit-learn gradio
    jupyter notebook

  Open Untitled1.ipynb and run the cells from top to bottom.
  On Windows, the Linux commands "!ls" and "ls -lh ..." in the
  notebook do not work. Delete those cells or replace them with
  "!dir".

WHAT EACH PART OF THE NOTEBOOK DOES

  Step 1 - Setup
      Installs libraries and clones the seq2point-nilm reference
      repository.

  Step 2 - Simulate training data
      Creates 7 days of 1-minute data: an AC (random 1500-2500 W, on
      between 12:00 and 18:00), a random base load and noise.
      Output: train_data_ac.csv

  Step 3 - Build windows
      Cuts the aggregate signal into sliding windows of 599 minutes.
      Each window is used to predict the AC power at its midpoint.

  Step 4 - Train the model
      Trains a Seq2Point convolutional network (5 Conv1D layers, a
      dense layer, dropout) for 5 epochs.
      Output: saved_models/seq2point_ac1.keras

  Step 5 - Test on unseen data
      Simulates a new "mystery" day with different random settings,
      predicts the AC and plots actual against predicted.
      Output: prediction_results.png

  Step 6 - Dashboard 1
      Single-appliance (AC) dashboard. Interactive Gradio app.

  Step 7 - Dashboard 2
      Multi-appliance dashboard. Interactive Gradio app.

IMPORTANT: Run the training cell BEFORE the dashboards. Both
dashboards load ./saved_models/seq2point_ac1.keras. Dashboard 1 stops
with "Model file not found" if it is missing.

NOTE ON THE EARLY CELLS: The cells that call seq2point_train.py,
train_main.py and custom_train.py are early experiments with the
cloned reference repository. The final results do not use them,
because the notebook later trains its own Keras model. If one of them
gives an error, skip it and continue.


---------------------------------------------------------------------
 DASHBOARD FEATURES
---------------------------------------------------------------------

Both dashboards work the same way: click the button and a new random
household day is simulated, passed through the model, and summarised.
Every click gives a different day.

DASHBOARD 1 - Household Smart Grid & Appliance Disaggregation
(single appliance)

Button: "Simulate New Raw Household Data & Disaggregate"

  Randomised day
      New AC power (1600-2400 W), base load (250-450 W), AC start hour
      (11-14) and run length (4-7 hours) on every click.

  Total Daily Energy (kWh)
      Sum of the aggregate household reading over the day.

  Total Daily Bill Estimate
      Total energy x tariff of $0.15 per kWh.

  Extracted AC Energy (kWh)
      AC energy estimated by the neural network from the aggregate
      signal only.

  AC Consumption Cost
      Cost of the AC energy, with its percentage of the total bill.

  Live Power Usage Timeline
      Grey line: total household power. Blue line: real AC power
      (ground truth). Red dashed line: AC power predicted by the
      model, so you can see how closely they agree.

  Insights panel
      Text summary: when the AC ran, what share of the day's energy it
      used, and an estimated saving from shifting its use to earlier
      hours.

DASHBOARD 2 - Interactive Multi-Appliance Smart Home Dashboard

Button: "Simulate Multi-Appliance Daily Telemetry"

  Five appliances
      AC, Fridge, Washing Machine, Kettle and Microwave, each with
      randomised power and schedule.

  Two-panel timeline
      Top: the actual power of every appliance together with the
      total. Bottom: the extracted (predicted) power of every
      appliance, in the same colours, so the two panels can be
      compared.

  Appliance Metric Ledger
      Table of estimated energy (kWh) and cost ($) for each
      appliance.

  Multi-Appliance Audit panel
      Text summary of total energy and cost, the high-consumption
      devices (AC, washing machine), the combined energy of the short
      heating spikes (kettle and microwave), and advice about running
      heating appliances at the same time.


---------------------------------------------------------------------
 CURRENT SCOPE AND LIMITATIONS
---------------------------------------------------------------------

Please read these before presenting or relying on the results.

  * All data is simulated. The model has not been trained or tested
    on real household meter data.

  * Only the AC is predicted by a trained neural network. In
    Dashboard 2, the fridge, washing machine, kettle and microwave
    "extracted" values are the simulated true values plus random
    noise. They illustrate the intended output and are NOT model
    predictions yet.

  * The "LLM" insight panels are template text. They fill in numbers
    from the simulation using fixed sentences. No language model is
    called. The savings figure (25% of the AC cost) and the 19:40
    kettle/microwave advice are fixed assumptions, not conclusions
    drawn from the data.

  * The training scenario is narrow. The model sees one AC with a
    single block of use per day. Its accuracy on other patterns
    (cycling, several appliances at once, unknown appliances) has not
    been measured.

  * Fixed tariff. Costs use $0.15 per kWh everywhere.

  * Short training. The model is trained for 5 epochs on 7 days of
    data, which suits a demo, not production use.

Natural next steps: train one model per appliance on a richer
simulator (or on a public dataset such as iAWE, REDD or UK-DALE), add
detection of unknown appliance signatures, report an error metric for
each appliance, and compare the sum of estimated appliance energy with
the measured total.


---------------------------------------------------------------------
 TROUBLESHOOTING
---------------------------------------------------------------------

Problem: FileNotFoundError: Model file not found
  Fix: Run the training cell first so saved_models/seq2point_ac1.keras
       exists.

Problem: pip install tensorflow fails
  Fix: Use Google Colab, or a virtual environment with an older
       Python version.

Problem: The Gradio panel does not show
  Fix: Re-run the dashboard cell. In Colab, make sure the earlier
       "pip install gradio" cell ran.

Problem: "ls" is not recognised (Windows)
  Fix: Remove that cell or use "dir" instead.

Problem: A cell using seq2point_train.py or custom_train.py fails
  Fix: Skip it. The final model is trained by the cell that builds
       the Keras Sequential model.

Problem: Results differ each run
  Fix: Expected. Every run uses random power levels and schedules.


---------------------------------------------------------------------
 CREDITS
---------------------------------------------------------------------

  * Seq2Point approach for NILM: Zhang et al., "Sequence-to-point
    learning with neural networks for non-intrusive load monitoring"
    (AAAI 2018).
  * Reference repository cloned by the notebook:
    https://github.com/MingjunZhong/seq2point-nilm
  * Built with TensorFlow/Keras, Gradio, pandas, NumPy and
    Matplotlib.
