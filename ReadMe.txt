"Target Response Adaptation for Correlation Filter Tracking" ECCV2016

Authors: Adel Bibi, Matthias Mueller, and Bernard Ghanem.

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


The code is integratable with the OTB100 and OTB50 evaulation benchmarks.
To run the code over the complete benchmark:

1- Move the complete traker directory to the "Trackers" directory in the OTB evulation code.
Locate the function "configTrackers.m" in the OTB100 evaulation code. To install the OTB100 benchmark:
[1] http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html
[2] https://sites.google.com/site/trackerbenchmark/benchmarks/v10

2- Add the following line to the list of trackers to be evualted over:
struct('name','SAMF_AT','namePaper','SAMF_AT')
Note: The code that will be run through the evaulation by running the function "run_SAMF_AT.m".

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
