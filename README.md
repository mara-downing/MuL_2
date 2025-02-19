# MuL\_2

Each experiments folder holds a different set of robustness experiments from the paper. Within each folder, there will be a set of subfolders corresponding to each network under analysis. Within those folders, there will be the python code files as well as folders starting with "output" which contain the experimental data. Each output folder is marked with the robustness radius used, the time limit, and then a set of letters:

s or b: sphere or ball sampling

u or a: C or CA (respectively) bounds handling

u or b: uniform or balanced weighting by complete bounds

n, d, s, or a: whether to reject no adjacencies, different classified adjacencies, same classified adjacencies, or all adjacencies

n or y: only use n


Output folders contain a file called summary.txt with the confusion matrix. The progam also outputs individual files for each input, but due to github file size requirements these could not be included.

In any folder with a network.py file set up, experiments can be run by:

`main.py <inputsfile> <r> <time_limit_seconds> <s or b> <u or a> <u or b> <n, d, s, or a> n`

The Dockerfile is provided for easy environment setup; it contains commented commands at the bottom for building and running. The run command should be executed in the MuL2 directory.

The main folder also contains a set of "template" python files---the main.py file is ready with all of the sampling code, and the other files are empty functions user code to read inputs, execute the network, and add feasability bounds.

For the TSRD networks, we provide the code used to generate the networks in the training folder. Datasets and networks from tools we used in comparison must be obtained from their source.

Lenet1, Lenet5, and ResNet used in Tables I, II, and III are obtained from the DeepHunter artifact (linked below). ResNet50 and Inception-V3 for ImageNet are obtained from the python torch library.

Correctly classified inputs files are omitted for the ImageNet networks due to file size limitations.

Results in the concolic_comp folder (comparison with DeepConcolic) are split into 5 subfolders---as our approach does not need any saved state, we summed results across these 5 2 hour experiments to form one 10 hour experiment. This split was necessary due to a memory leak in Tensorflow v2.14.0.

The DeepHunter tool is obtained from https://bitbucket.org/xiaofeixie/deephunter/src/v3/, the DeepConcolic tool is obtained from the url listed in the Sun et al. 2018 paper, and the DeepRover tool is obtained from the url listed in the Zhang et al. 2023 paper.
