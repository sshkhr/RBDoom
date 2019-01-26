RBDoom
=======
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)


RBDoom is a RAINBOW [[1]](#references) based agent for playing the first-person shooter game Doom using the VizDoom platform. RAINBOW cmobines various improvements in Deep-Q Learning and demonstrated better overall performance. It is made up of seven components:

1. DQN [[2]](#references)
2. Double DQN [[3]](#references)
3. Prioritised Experience Replay [[4]](#references)
4. Dueling Network Architecture [[5]](#references)
5. Multi-step Bootstrap Returns [[6]](#references)
6. Distributional RL [[7]](#references)
7. Noisy Nets [[8]](#references)

Requirements
------------

- [VizDoom](http://vizdoom.cs.put.edu.pl/)
- [Scikit Image](http://scikit-image.org/docs/dev/api/skimage.html)
- [PyTorch](http://pytorch.org/)

Acknowledgements
----------------

- [@Kaixin](https://github.com/Kaixhin) for [Rainbow](https://github.com/Kaixhin/Rainbow)
- [@jaara](https://github.com/jaara) for [AI-blog](https://github.com/jaara/AI-blog)

References
----------

[1] [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)  
[2] [Playing Atari with Deep Reinforcement Learning](http://arxiv.org/abs/1312.5602)  
[3] [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)  
[4] [Prioritized Experience Replay](http://arxiv.org/abs/1511.05952)  
[5] [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)  
[6] [Reinforcement Learning: An Introduction](http://www.incompleteideas.net/sutton/book/ebook/the-book.html)  
[7] [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)  
[8] [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)  
