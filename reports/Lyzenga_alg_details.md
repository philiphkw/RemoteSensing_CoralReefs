[WORK IN PROGRESS]

For this study we will be using a slightly simplified version of the Lyzenga Algorithm. However, first it is important to understand how the algorithm works.

The goal of the algorithm is to identify and correct for quality degradation in bottom signals caused by the absorption and scattering of spectral bands in the water column. As light moves down the water column, the band's reflectance decays exponentially. Mathematically, this means that the relationship between band strength and depth is curved.

Without knowing the actual depth at each pixel, we cannot directly plot how reflectance changes with depth. However, depth affects all bands simultaneously — as a pixel gets deeper, every band's reflectance drops together. This means that if we plot two bands against each other, their values will move together in a predictable way as depth changes. A shallow sand pixel will have high reflectance in both bands. The same sand pixel at greater depth will have lower
reflectance in both bands. Plotting one band against the other traces out this joint variation, and it is that joint variation that encodes depth — even without ever measuring depth directly. So for this to properly work, the pixels used to calibrate the algorithm have to capture the same bottom type at different depths so that the co-variance can properly isolate the effect of the water column. If the pixels contain a mixture of bottom type, the algorithm can no longer distinguish how much of the change in reflectance is caused by the water column and how much is caused by the change bottom type. 

Now that we have some understanding of the background, we can start delving into the mathematics. From now on, band one will be mathematically represented as `xi`, and band two as `xj`. After keeping bottom type constant, the problem of an exponential relationship remains. Although not impossible to derive depth from this curved relationship, linearising it makes it computationally much more simple to solve. To do this both `xi` and `xj` are log transformed into `Xi` and `Xj`, and when we plot these new log values again, we will get a linear line and a slope (`ki/kj`) that now represents the direction of depth; lower values corresponds with a deeper depth, and vice versa. 

<img src="figures\image-1.png" width="800">

So far we've only talked about plotting the reflectance values of two bands for only one bottom type. If we then plot multiple bottom types on the same graph after log-transformation (using the same two bands) we will see multiple parallel lines next to each other; each line representing a different bottom type. These lines are parallel because, as we discussed before, depth affects all bands in the same way, so the slope (`ki/kj`) for each reflectance value will be identical.

<img src="figures\image-2.png" width="800">
<!-- <img src="image.png" width="400"> -->

The next problem is how to cancel out depth so that a pixel's measured reflectance only tells you about its bottom type, not how deep it is. In the log-transformed plot, the parallel lines all run in the same direction: the depth direction. But our measurement axes `Xi` and `Xj` cut diagonally across those lines, so every observation mixes depth and bottom type together.

The fix is to rotate the axes until one axis points exactly along the lines (the depth direction) and the other points perpendicular to them (across the lines). After this rotation:

- Moving along the first axis changes depth but keeps you on the same line (same bottom type).
- Moving along the second axis crosses from one line to another (different bottom types) while depth stays constant.

That second axis is the "depth-invariant index". Any pixel's value on that axis tells you purely about its bottom type, with depth's influence removed.


<img src="figures\image-3.png" width="800">

<!-- To help you understand this better I will provide an example. It's the same as having a perfectly vertical line and a perfectly horizontal line on a graph. These lines are also perpendicular to one another and by moving across one line the value of the other line will stay the same. Similarly to longitudes and latitudes on a map. If you move straight along the longitude, your latitude value will stay the same, and vice versa. In our case, a  -->

Sources:
- https://www.researchgate.net/profile/David-Lyzenga/publication/41511121_Passive_remote_sensing_techniques_for_mapping_water_depth_and_bottom_features/links/0deec53298d903438a000000/Passive-remote-sensing-techniques-for-mapping-water-depth-and-bottom-features.pdf
- https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-supported-plugins/org.esa.sen2coral.sen2coral.algorithms/depthInvariantIndices/DepthInvariantIndicesAlgoSpec.html