# SfM Point Cloud Generation

## Final Update

Hardware has successfully been able to return inputs for all keypoints. However, these values are not very accurate,
likely due to the fixed-point, half-precision storage used for mathematics in the PL. Future improvements could include
Increasing the bit-width of the histogram tensor to 32-bits, still fixed point. This may give us more precision. Potentially
exploring more effective storage methods that would allow us to partition the array could improve the latency at the cost of
a massive increase in board area.
