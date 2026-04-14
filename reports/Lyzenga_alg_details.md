For this study we will be using a slightly simplified version of the Lyzenga Algorithm. However, first it is important to understand how the algorithm works (if you already know how it works you can skip to the 'Simplification' section). In order to understand the lyzenga.py script, we recommend you to read the whole document. 

>NOTE: This version of the algorithm is inspired by the original [Lyzenga 1978 paper](https://doi.org/10.1364/AO.17.000379), a general [depth-invariance guide](https://step.esa.int/main/wp-content/help/versions/9.0.0/snap-supported-plugins/org.esa.sen2coral.sen2coral.algorithms/depthInvariantIndices/DepthInvariantIndicesAlgoSpec.html), and [Teongu's github repository](https://github.com/teongu/lyzenga1978/) explaining how to apply the Lyzenga Algorithm in python.

## Background
The goal of the algorithm is to identify and correct for quality degradation in bottom signals caused by the absorption and scattering of spectral bands in the water column. As light moves down the water column, the band's reflectance decays exponentially. Mathematically, this means that the relationship between band strength and depth is curved.

Without knowing the actual depth at each pixel, we cannot directly plot how reflectance changes with depth. However, depth affects all bands simultaneously — as a pixel gets deeper, every band's reflectance drops together. This means that if we plot two bands against each other, their values will move together in a predictable way as depth changes. A shallow sand pixel will have high reflectance in both bands. The same sand pixel at greater depth will have lower
reflectance in both bands. Plotting one band against the other traces out this joint variation, and it is that joint variation that encodes depth — even without ever measuring depth directly. So for this to properly work, the researcher has to manually select pixels that contain the same bottom type at different depths so that the co-variance can properly isolate the effect of the water column. If the pixels contain a mixture of bottom type, the algorithm can no longer distinguish how much of the change in reflectance is caused by the water column and how much is caused by the change bottom type. 

## Mathematics
Now that we have some understanding of the background, we can start delving into the mathematics. From now on, band one will be mathematically represented as `xi`, and band two as `xj`. After keeping bottom type constant, the problem of an exponential relationship remains. Although not impossible to derive depth from this curved relationship, linearising it makes it computationally much more simple to solve. To do this both `xi` and `xj` are log transformed into `log(xi) = Xi` and `log(xj) = Xj`, and when we plot these new log values again, we will get a linear line and a slope (`ki/kj`) that now represents the direction of depth; lower values corresponds with a deeper depth, and vice versa. 

<img src="figures\image-1.png" width="800">

So far we've only talked about plotting the reflectance values of two bands for only one bottom type. If we then plot multiple bottom types on the same graph after log-transformation (using the same two bands) we will see multiple parallel lines next to each other; each line representing a different bottom type. These lines are parallel because, as we discussed before, depth affects all bands in the same way, so the slope (`ki/kj`) for each reflectance value will be identical.

<img src="figures\image-2.png" width="800">

The next problem is how to cancel out depth so that a pixel's measured reflectance only tells you about its bottom type, not how deep it is. In the log-transformed plot, the parallel lines all run in the same direction: the depth direction. But our measurement axes `Xi` and `Xj` cut diagonally across those lines, so every observation mixes depth and bottom type together.

The fix is to rotate the axes until one axis points exactly along the lines (the depth direction) and the other points perpendicular to them (across the lines). After this rotation:

- Moving along the x-axis changes depth but keeps you on the same line (same bottom type).
- Moving along the y-axis crosses from one line to another (different bottom types) while depth stays constant.

The y-axis is the "depth-invariant index". Any pixel's value on that axis tells you purely about its bottom type, with depth's influence removed. The values along the y-aixs no longer literally represent reflectance but rather an arbitrary measure of distance (similar to how centimeters on a ruler are also an arbitrary distance measure).

<img src="figures\image-3.png" width="800">

<br>

Mathematically, the Depth Invariant Index (DII) for two bands is calculated as:
$$DII_{ij} = X_i - (k_i/k_j) * X_j$$

Where:
- $X_{i/j}$: The log transformed raw band
- $k_i/k_j$: The attenuation ratio

This new value $DII_{ij}$ is computed for every pixel of a given band creating a new band where depth has been removed as factor.

To make matters more complex, when working with more than two bands you now have to work in an $N$-dimensional space rather than a 2D plot. Instead of plotting two log-transformed bands ($X_i$ vs $X_j$​) on a 2D graph, imagine plotting $N$ log-transformed bands in an $N$-dimensional space. In this space each pixel is represented as a point with $N$ number of coordinates. All pixels with the same bottom type form a hyperplane. All hyperplanes are parallel to each other (because depth affects all bands identically). And the direction perpendicular to these hyperplanes is the depth direction.

Rather than rotating axes in 2D, you now need to identify the depth direction in this $N$-dimensional space and project away from it. The result is a set of depth-invariant indices, one for each pair of bands you can extract from your $N$ bands.

## Algorithm Simplification
For this research we simplify the lyzenga algorithm in two ways:

1. We use a Digital Elevation Model (DEM) rather than using the covariance method becuase that is only necessary if depth is not already given (which in most cases it isn't).
2. We reduce our band selection to only the three bands that are good in penetrating water: Coastal Blue, Blue, and Green. It's also why we don't use a matrix, because it's computationally overkill for only three bands.

In our case, the area we are researching already contains depth data. Therefore, instead of _manually_ selecting pixels with the same bottom type to help the algorithm plot a regression, we can simply estimate each band's attenuation coefficient by regressing the band values of _all_ pixels against their corresponding depth values. To find the depth values for each pixel, we only have to overlap the DEM raster with the satellite raster by matching their AOI and resolution.

Another unique feature to our application of the Lyzenga Algorithm is that we don't use a matrix to calculate the attenuation ratio for all bands simultaneously. Rather, we chose to manually calculate the ratio for every band combination of the three bands. This results in three depth-invariant bands: CB-Blue, CB-Green, Blue-Green. We do this because:

1. The results are easier to interpret
2. It's less computationally expensive, and 
3. It gives us flexibility in choosing which pairs should be kept or discarded (due to poor calibration), or use them in downstream processes such as replacing the Blue-Green ratio index with the Blue-Green depth invariance index (because they capture the same thing), or calculating brightness (which is an average of the three depth-invariant bands).  
