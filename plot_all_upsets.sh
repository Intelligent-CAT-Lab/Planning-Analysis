#!/usr/bin/env bash

python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan no_plan
python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan no_reproduce
python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan no_validation
python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan plan_and_regression
python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan plan_and_summary
python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan plan_reminded
python lang_analysis/updset_plot.py --benchmark SWE-Bench-Verified plan plan_reordered
