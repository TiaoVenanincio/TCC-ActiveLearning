Deep Active Learning via Strategy Committees

Manual annotation remains a major bottleneck for training deep learning models in visual recognition tasks. Although Deep Active Learning reduces labeling effort, most prior studies evaluate acquisition criteria in isolation, leaving it unclear how complementary strategies should be systematically combined. This work addresses this limitation by investigating committee-based Deep Active Learning as a principled framework for unifying diversity- and uncertainty-driven sampling within a single, structured acquisition strategy. We combine Least Confidence, Margin Sampling, and Entropy Sampling with K-means-based diversity selection through explicit interaction schemes (sequential, intersection, and union) within a controlled incremental active learning protocol. Experiments on MNIST and FashionMNIST evaluate not only predictive accuracy but also class discovery dynamics and label correction behavior throughout the learning process. Results show that committee-based strategies match or surpass the best individual methods while accelerating class coverage and reducing annotation requirements. More importantly, we observe a consistent learning mechanism: diversity improves early representation coverage, whereas uncertainty-based criteria refine decision boundaries at later stages. The benefits increase with dataset complexity, transitioning from stability improvements on simpler datasets to measurable accuracy gains on more complex data. These findings provide empirical guidelines on when and how to combine acquisition criteria, highlighting the importance of balancing exploration and refinement for label-efficient deep active learning.

Citation

If you think our work is meaningfull in your reasearch, please cite us:
@inproceedings{Neto2026,
  author    = {Sebastião V. G. Neto and Daniel M. Galetti and Pedro H. Bugatti and Cid A. N. Santos and Willian P. Amorim and Priscila T. M. Saito},
  title     = {Deep Active Learning via Strategy Committees},
  booktitle = {25th International Conference on Artificial Intelligence and Soft Computing},
  pages     = {1—12},
  year      = {2026}
}

Acknowledgement

This research has been supported by grants from UFSCar and Instituto de Pesquisas Eldorado.
