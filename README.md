# Gaussian Splatting Co-design Project
## By Benjamin Taylor and Ariel Mahler

### Update regarding splatting portion

While Ben was able to create a working SW implementation of the SfM point cloud generation, investigation into the mathematics and hardware requirements of the rendering process itself has proven to be much more complex than previously
understood. That is, the paper that we have been referencing requires CUDA cores, the use of floating point mathematics, and a high-speed graphics card to produce its desired results of 1080p @ 60 fps (real-time) rendering.

With this in mind, our focus for the future updates will be to work on specifically refining the SfM implementation, which seems more promising for the FPGA board that we have. In the meantime, additional research will be done on the 
feasibility of a non-floating point, non-CUDA, implementation of the splatting rendering method.
