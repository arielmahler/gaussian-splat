# SfM Point Cloud Generation

## Final Update

Hardware has successfully been able to return inputs for all keypoints. However, these values are not very accurate,
likely due to the fixed-point, half-precision storage used for mathematics in the PL. Future improvements could include
Increasing the bit-width of the histogram tensor to 32-bits, still fixed point. This may give us more precision. Potentially
exploring more effective storage methods that would allow us to partition the array could improve the latency at the cost of
a massive increase in board area.

## Running the Program:

To run the program, we recommend cloning the github repository, which can be found here:
https://github.com/arielmahler/gaussian-splat

The `main` or `final_update` branches can be used.

The bitfiles located in the repository must be moved to the `\home\xilinx\pynq\ovelays\` folder under a new
folder named `descriptor`. Ensure that all bitfiles are named `design_1`.

Once this is done, place the `dino_data` folder into the same folder as the `sfm_algo_unpacked` python file
and the `final_project_update` jupyter notebook. It is important that these three items are in the same
folder.

From here, running the Jupyter notebook in order should return an image with the keypoints on them.
