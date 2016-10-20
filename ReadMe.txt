"Target Response Adaptation for Correlation Filter Tracking" ECCV2016

Authors: Adel Bibi, Matthias Mueller and Bernard Ghanem.

Visit our group's website: https://ivul.kaust.edu.sa/Pages/Home.aspx

Adel Bibi's website: www.adelbibi.com

Email:

   adel.bibi [AT] kaust.edu.sa                 bibiadel93 [AT] gmail.com
   matthias.mueller.2 [AT] kaust.edu.sa
   Bernard.Ghanem [AT] kaust.edu.sa 

Website:
   Adel Bibi: www.adelbibi.com
   Bernard Ghanem: http://www.bernardghanem.com/
   IVUL: https://ivul.kaust.edu.sa/Pages/Home.aspx



Please cite:

@inproceedings{bibi2016target,
  title={Target response adaptation for correlation filter tracking},
  author={Bibi, Adel and Mueller, Matthias and Ghanem, Bernard},
  booktitle={European Conference on Computer Vision},
  pages={419--433},
  year={2016},
  organization={Springer}
}

**************************************************
This is a MATLAB implementation on the adaptive target for correlation filters.
The framework is generic and can be directly implemented in any correlation tracker that
solves the following objective. ||Ax - b||_2^2 + \lambda ||x||_2^2.

* The code is based on the tracker SAMF [1].

It is free for research use. If you find it useful, please acknowledge the paper
above with a reference.
**************************************************


To run over 1 single video:

1- Open ('run_tracker.m').
2- Change line 49 video = 'choose'.
3- Dump any video of OTB100/OTB50 into the the directory (videos).
4- Run the tracker.

The annotation files and the attributes for all the videos of OTB100 are avilable in the directory (anno).


To run over the complete dataset:

1- Open ('run_tracker.m').
2- Change line line 49 video = 'all'.
3- Dump all the OTB100 or OTB50 videos into the the directory (videos).
4- Run the tracker.
5- The detailed results will be stored in (results) directory. (Make sure to create a directory named results in the current path)


The code is also completey integratable with the OTB100 and OTB50 evaulation benchmarks.
To do so:

1- Move the complete traker directory to the (Trackers directory in the OTB evulation code).
The function is called (configTrackers.m) in the OTB evaulation code. It can be found here:
[1] http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
[2] https://sites.google.com/site/trackerbenchmark/benchmarks/v10

2- Add the following line to the list of trackers to be evualted over:
struct('name','SAMF_AT','namePaper','SAMF_AT')
Note: The code that will be run through the evaulation through (run_SAMF_AT.m).
It's the same exact code with the same parameters but has been changed to the standard OTB format.


**************************************************

Referrences:
[1] A Scale Adaptive Kernel Correlation Filter Tracker with Feature Integration. European Conference on Computer Vision Workshops 2014.
[2] Henriques, João F., et al. "High-speed tracking with kernelized correlation filters." Pattern Analysis and Machine Intelligence, IEEE Transactions on 37.3 (2015): 583-596.
[3] Henriques, Joao F., et al. "Exploiting the circulant structure of tracking-by-detection with kernels." Computer Vision–ECCV 2012. Springer Berlin Heidelberg, 2012. 702-715.
[4] Wu, Yi, Jongwoo Lim, and Ming-Hsuan Yang. "Online object tracking: A benchmark." Proceedings of the IEEE conference on computer vision and pattern recognition. 2013.
A complete list of references can be found in the paper, which can be found here
https://ivul.kaust.edu.sa/Pages/Pub-Adaptive-Kernelized-Correlation-Filters.aspx

Paper:
http://www.adelbibi.com/papers/ECCV2016/Target_Adap.pdf

Supplemental Material:
[1] http://www.adelbibi.com/papers/ECCV2016/Target_Adap_supp.pdf
[2] https://www.youtube.com/watch?v=yZVY_Evxm3I
