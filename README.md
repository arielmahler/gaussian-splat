# SfM Pointcloud generation
## Updates for March 28th

Our main goal for these previous weeks was to time functions as they ran in python, to better understand the things that would benefit most from being moved to hardware. There is a large function called `generateDescriptors`, which is the function that takes the longest to run. This function takes a large array of 4 zipped values, then operates many identical math functions to them and an array representing floating point values. Because this is repeated a ton of times, this is ideal to implement in hardware. Unfortunately, due to time constraints and lack of testing time, we were unable to produce a fully synthesizable c++ program to deliver this week. However, the prototype/pseudocode we created this week can be found in `GenerateDescriptor_impl.cpp`. This includes the basic information and typing that we expect we will need for the math functions. We suspect that from here it is just a question of ironing out details regarding types and the storage of the tensor array, which will have to be accessed across loops.