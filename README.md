# DiamFit
Fits a diameter and limb-darkening coefficient to interferometric visibilities and/or closure amplitudes.
Linear or power law limb darkening laws are both available. 
When fitting to visibilities, fits a scaling factor for each night+combiner. Uncertainties calculated by a block-bootstrap method.
The blocks are determined by combiner, baseline, and time. So, for example, three observations made with both mircx and mystic using all six CHARA telescopes would give 90 blocks and each block would be made up of N visibilities (where N is the number of wavelength channels for the observation).
The bootstrapping algorithm selects a random set of blocks (with replacement). Each block within the selection has its visibility uniformly varied by a random gaussian value based on the average uncertainty of the block. The diameter is then fit to this random set. The uncertainty in the measured values is determined by running this algorithm many times (1000 times by default).

Required Inputs
* data_dir - the relative path to the directory with the reduced+calibrated oifits that you want to use are located
* stars - a star or list of stars to fit
* diam_init - the initial guess for the diameter
* ldcK_init - the initial guess for the K-band limb-darkening coefficient
* ldcH_init - the initial guess for the H-band limb-darkening coefficient
* combiners - a beam combiner (or list of combiners) your data come from
* nights - the night or list of nights your data come from

**IMPORTANT NOTE - Your oifits files must be named in the following format to be recognized by diam_fit.py**

STAR.COMBINER.NIGHT*.oifits

Examples:
* altair.MIRCX.2026Jun16.oifits
* vega.MYSTIC.2025Oct03.01.oifits
* vega.MYSTIC.2025Oct03.02.oifits
* HD_27371.PIONIER.2027Jul01.extra_label.oifits

When you call diam_fit, the star, combiner, and nights must match the values in the filenames or else the files won't be read

Optional Inputs
* scenarios - accepted values are ['by_file','by_night','by_combiner','all'], 'all' is default. Multiple options allowed
* law - accepted values are ['power','linear'], 'power' is default
* boot_N - number of iterations for the bootstrapping. Default is 1000
* fix_ldcK - fix ldcK to ldcK_init (default = false)
* fix_ldcH - fix ldcH to ldcH_init (default = false)
* camp - fit closure amplitudes (default = false)

**Example runs**

Multistar:

  python diam_fit.py --data_dir final_data_2023Dec --stars HD_27371 HD_27697 HD_28305 HD_28307 --diam_init 1.5 1.5 1.5 1.5 --ldcK_init 0.3 0.3 0.3 0.3 --ldcH_init 0.3 0.3 0.3 0.3 --combiners MIRCX MYSTIC --nights 2023Dec12 2023Dec13

Single star:

  python diam_fit.py --data_dir final_data_2023Dec --stars HD_27371 --diam_init 1.5 --ldcK_init 0.3 --ldcH_init 0.3 --combiners MIRCX MYSTIC --nights 2023Dec12 2023Dec13


**ANOTHER IMPORTANT NOTE - You will need a list of time windows for your observations as part of the block-bootstrapping. This is not measured automatically. See MJD_cutoffs.csv as an example. This file must be in your data directory.**
