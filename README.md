# deep-learning-content-moderation

Various sources for deep learning based content moderation, sensitive content detection, scene genre classification, nudity detection, violence detection, substance detection from text, audio, video &amp; image input modalities.

## citation

If you find this source useful, please consider citing it in your work as:

```bib
@INPROCEEDINGS{10193621,
  author={Akyon, Fatih Cagatay and Temizel, Alptekin},
  booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)}, 
  title={State-of-the-Art in Nudity Classification: A Comparative Analysis}, 
  year={2023},
  pages={1-5},
  keywords={Analytical models;Convolution;Conferences;Transfer learning;Benchmark testing;Transformers;Safety;content moderation;nudity detection;safety;transformers},
  doi={10.1109/ICASSPW59220.2023.10193621}}
```
```bib
@article{akyon2022contentmoderation,
  title={Deep Architectures for Content Moderation and Movie Content Rating},
  author={Akyon, Fatih Cagatay and Temizel, Alptekin},
  journal={arXiv},
  doi={https://doi.org/10.48550/arXiv.2212.04533},
  year={2022}
}
```

## table of contents

* [datasets](https://github.com/fcakyon/content-moderation-deep-learning#datasets)
    * [movie and content moderation datasets](https://github.com/fcakyon/content-moderation-deep-learning#movie-and-content-moderation-datasets)
* [techniques](https://github.com/fcakyon/content-moderation-deep-learning#techniques)
    * [sensitive content detection](https://github.com/fcakyon/content-moderation-deep-learning#sensitive-content-detection)
        * [movie content rating](https://github.com/fcakyon/content-moderation-deep-learning#movie-content-rating)
        * [content moderation](https://github.com/fcakyon/content-moderation-deep-learning#content-moderation)
    * [movie/scene genre classification](https://github.com/fcakyon/content-moderation-deep-learning#movie/scene-genre-classification)
    * [multimodal architectures](https://github.com/fcakyon/content-moderation-deep-learning#multimodal-architectures)
        * [synchronous multimodal architectures](https://github.com/fcakyon/content-moderation-deep-learning#synchronous-multimodal-architectures)
        * [asynchronous multimodal architectures](https://github.com/fcakyon/content-moderation-deep-learning#asynchronous-multimodal-architectures)
    * [action recognition](https://github.com/fcakyon/content-moderation-deep-learning#action-recognition)
        * [with transformers](https://github.com/fcakyon/content-moderation-deep-learning#with-transformers)
        * [with 3D CNNs](https://github.com/fcakyon/content-moderation-deep-learning#with-3d-cnns)
    * [contrastive representation learning](https://github.com/fcakyon/content-moderation-deep-learning#contrastive-representation-learning)
    * [review papers](https://github.com/fcakyon/content-moderation-deep-learning#review-papers)
* [tools](https://github.com/fcakyon/content-moderation-deep-learning#tools)

## datasets

### movie and content moderation datasets

| name | paper | year | url | input modality | task | labels |
|--- |--- |--- |--- |--- |--- |--- |
| LSPD | [pdf](https://inass.org/wp-content/uploads/2021/09/2022022819-4.pdf) | 2022 | [page](https://sites.google.com/uit.edu.vn/LSPD) | image, video | image/video classification, instance segmentation | porn, normal, sexy, hentai, drawings, female/male genital, female breast, anus |
| MM-Trailer | [pdf](https://aclanthology.org/2021.ranlp-1.146.pdf) | 2021 | [page](https://ritual.uh.edu/RANLP2021/) | video | video classification | age rating |
| Movienet | [scholar](https://scholar.google.com/scholar?cluster=7273702520677604457&hl=en&as_sdt=0,5) | 2021 | [page](https://movienet.github.io/) | image, video, text | object detection, video classification | scene level actions and places, character bboxes |
| Movie script severity dataset | [pdf](https://arxiv.org/pdf/2109.09276.pdf) | 2021 | [github](https://github.com/RiTUAL-UH/Predicting-Severity-in-Movie-Scripts) | text | text classification | frightening, mild, moderate, severe |
| LVU | [pdf](https://chaoyuan.org/papers/lvu.pdf) | 2021 | [page](https://chaoyuan.org/lvu/) | video | video classification | relationship, place, like ration, view count, genre, writer, year per movie scene |
| Violence detection dataset | [scholar](https://scholar.google.com/scholar?cluster=839991967738597651&hl=en&as_sdt=0,5) | 2020 | [github](https://github.com/airtlab/A-Dataset-for-Automatic-Violence-Detection-in-Videos) | video | video classification | violent, not-violent |
| Movie script dataset | [pdf](https://sail.usc.edu/~mica/pubs/martinez2019violence.pdf) | 2019 | [github](https://github.com/usc-sail/mica-violence-ratings) | text | text classification | violent or not |
| Nudenet | [github](https://github.com/notAI-tech/NudeNet) | 2019 | [archive.org](https://archive.org/details/NudeNet_classifier_dataset_v1) | image | image classification | nude or not |
| Adult content dataset | [pdf](https://arxiv.org/abs/1612.09506) | 2017 | [contact](https://drive.google.com/file/d/1MrzzFxQ9t56i9d3gMAI3SaHXfVd1UUDt/view) | image | image classification | nude or not |
| Substance use dataset | [pdf](https://web.cs.ucdavis.edu/~hpirsiav/papers/substance_ictai17.pdf) | 2017 | [first author](https://www.linkedin.com/in/roy-arpita/) | image | image classification | drug related or not |
| NDPI2k dataset | [pdf](https://www.researchgate.net/profile/Sandra-Avila-5/publication/308398120_Pornography_Classification_The_Hidden_Clues_in_Video_Space-Time/links/5a205361a6fdcccd30e00d4a/Pornography-Classification-The-Hidden-Clues-in-Video-Space-Time.pdf) | 2016 | [contact](https://www.ic.unicamp.br/~rocha/mail/index.html) | video | video classification | porn or not |
| Violent Scenes Dataset | [springer](https://link.springer.com/article/10.1007/s11042-014-1984-4) | 2014 | [page](https://www.interdigital.com/data_sets/violent-scenes-dataset) | video | video classification | blood, fire, gun, gore, fight |
| VSD2014 | [pdf](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_cbmi_2015.pdf) | 2014 | [download](http://www.cp.jku.at/datasets/VSD2014/VSD2014.zip) | video | video classification | blood, fire, gun, gore, fight |
| AIIA-PID4 | [pdf](https://www.igi-global.com/gateway/article/full-text-pdf/79140&riu=true) | 2013 | - | image | image classification | bikini, porn, skin, non-skin |
| NDPI800 dataset | [scholar](https://scholar.google.com/scholar?cluster=7836593192753784698&hl=en&as_sdt=0,5) | 2013 | [page](https://sites.google.com/site/pornographydatabase/) | video | video classification | porn or not |
| HMDB-51 | [scholar](https://scholar.google.com/scholar?cluster=5533751878557083824&hl=en&as_sdt=0,5) | 2011 | [page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#overview) | video |video classification | smoke, drink |

## techniques

### sensitive content detection

#### movie content rating

| name | paper | year | model | features | datasets | tasks | context |
|--- |--- |--- |--- |--- |--- |--- |--- |
| Movies2Scenes: Learning Scene Representations Using Movie Similarities | [scholar](https://scholar.google.com/scholar?cluster=10458780403101083615&hl=en&as_sdt=0,5) | 2022 | ViT-like video encoder + MLP | ViT-like video encoder embedings | Private, Movienet, LVU | movie scene representation learning, video classifcation (sex, violence, drug-use) | movie scene content rating |
| Detection and Classification of Sensitive Audio-Visual Content for Automated Film Censorship and Rating | [pdf](https://www.researchgate.net/profile/Mahmudul-Haque-4/publication/359704298_Detection_and_Classification_of_Sensitive_Audio-Visual_Content_for_Automated_Film_Censorship_and_Rating/links/6249a3aa8068956f3c668be5/Detection-and-Classification-of-Sensitive-Audio-Visual-Content-for-Automated-Film-Censorship-and-Rating.pdf) | 2022 | CNN + GRU + MLP | CNN embeddings from video frames | Violence detection dataset | violent/non-violent classification from videos | movie scene content rating |
| Automatic parental guide ratings for short movies | [page](http://eprints.utar.edu.my/4250/) | 2021 | separate model for each task: concat + LSTM, object detector, one-class CNN embeddings | video frame pixel values, image embeddings, text | Nudenet, private dataset | profanity, violence, nudity, drug classification | movie content rating |
| From None to Severe: Predicting Severity in Movie Scripts | [scholar](https://scholar.google.com/scholar?cluster=13391231771783888164&hl=en&as_sdt=0,5) | 2021 | multi-task pairwise ranking-classification network | GloVe, Bert and TextCNN text embeddings | Movie script severity dataset | rating classifcation (frightening, mild, moderate, severe) | movie content rating |
| A Case Study of Deep Learning-Based Multi-Modal Methods for Labeling the Presence of Questionable Content in Movie Trailers | [scholar](https://scholar.google.com/scholar?cluster=15170598737548453750&hl=en&as_sdt=0,5) | 2021 | multi-modal + multi output concat+MLP | CNN+LSTM video features, Bert and DeepMoji text embeddings, MFCC audio features | MM-Trailer | rating classifcation (red, yellow, green) | movie trailer content rating |
| Automatic Parental Guide Scene Classification Menggunakan Metode Deep Convolutional Neural Network Dan Lstm | [scholar](https://scholar.google.com/scholar?cluster=2229349718067727385&hl=en&as_sdt=0,5) | 2020 | 3 CNN model for 3 modality, multi-label dataset | CNN video and audio embeddings, LSTM text (subitle) embeddings | private dataset | gore, nudity, drug, profanity classification from video and subtitle | movie scene content rating |
| Multimodal data fusion for sensitive scene localization | [scholar](https://scholar.google.com/scholar?cluster=16351890771413386481&hl=en&as_sdt=0,5) | 2019 | meta-learning with Naive Bayes, SVM | MFCC and  prosodic features from audio, HOG and TRoF features from images | Pornography-2k dataset, VSD2014 | violent and pornographic scene localization from video | movie scene content rating |
| A Deep Learning approach for the Motion Picture Content Rating | [scholar](https://scholar.google.com/scholar?cluster=14110689052300247206&hl=en&as_sdt=0,5) | 2019 | MLP + rule-based decision | InceptionV3 image embeddings | Violent Scenes Dataset, private dataset | violence (shooting, blood, fire, weapon) classification from video | movie scene content rating |
| Hybrid System for MPAA Ratings of Movie Clips Using Support Vector Machine | [springer](https://link.springer.com/chapter/10.1007/978-981-13-1595-4_45) | 2019 | SVM | DCT features from image | private dataset | movie content rating classification from images | movie content rating |
| Inappropriate scene detection in a video stream | [page](https://dspace.bracu.ac.bd/xmlui/handle/10361/9469) | 2017 | SVM classifier + Lenet image classifier + rules-based decision | HoG and CNN features for image | private dataset | image classification: no/mild/high violence, safe/unsafe/pornoghraphy | movie frame content rating |

#### content moderation

| name | paper | year | model | features | datasets | tasks | context |
|--- |--- |--- |--- |--- |--- |--- |--- |
| State-of-the-Art in Nudity Classification: A Comparative Analysis | [ieee](https://ieeexplore.ieee.org/document/10193621) | 2023 | CNN, Transformers | EfficientNet, ViT, ConvNeXT image embeddings | LSPD, Nudenet, NDPI2k | nudity classification from images | general content moderation |
| Reliable Decision from Multiple Subtasks through Threshold Optimization: Content Moderation in the Wild | [scholar](https://scholar.google.com/scholar?cluster=6441617450690688428&hl=en&as_sdt=0,5) | 2022 | novel threshold optimization tech. (TruSThresh) | prediction scores | UnSmile (Korean hatespeech dataset) | optimum threshold prediction | social media content moderation |
| On-Device Content Moderation | [scholar](https://scholar.google.com/scholar?cluster=5550217109642251291&hl=en&as_sdt=0,5) | 2021 | mobilenet v3 + SSD object detector | mobilenet v3 image embeddings | private dataset | object detection + nudity classification from images | on-device content moderation |
| Gore Classification and Censoring in Images | [scholar](https://scholar.google.com/scholar?cluster=1477214110429045154&hl=en&as_sdt=0,5) | 2021 | ensemble of CNN + MLP | mobilenet v2, densenent, vgg16 image embeddings | private dataset | gore classification from images | general content moderation |
| Automated Censoring of Cigarettes in Videos Using Deep Learning Techniques | [scholar](https://scholar.google.com/scholar?cluster=14277066392319361311&hl=en&as_sdt=0,5) | 2020 | CNN + MLP | inception v3 image embeddings | private dataset | cigarette classification from video | general content moderation |
| A Multimodal CNN-based Tool to Censure Inappropriate Video Scenes | [scholar](https://scholar.google.com/scholar?cluster=11897464348691991288&hl=en&as_sdt=0,5) | 2019 | CNN + SVM | InceptionV3 image embeddings, AudioVGG audio embeddings | private dataset | inappropriate (nudity+gore) classification from video | general video content moderation |
| A baseline for NSFW video detection in e-learning environments | [scholar](https://scholar.google.com/scholar?cluster=12765316397376090724&hl=en&as_sdt=0,5) | 2019 | concat + SVM, MLP | InceptionV3 image embeddings, AudioVGG audio embeddings | YouTube8M, NDPI, Cholec80 | nudity classification from video | e-learning content moderation |
| Bringing the kid back into youtube kids: Detecting inappropriate content on video streaming platforms | [scholar](https://scholar.google.com/scholar?cluster=17864610709056312820&hl=en&as_sdt=0,5) | 2019 | CNN + LSTM (late fusion) | CNN based encoder for image, video and audio spectrograms | private dataset | video classification: orignal, fake explicit, fake violent | social media content moderation |

### movie/scene genre classification

| name | paper | year | model | features | datasets | tasks |
|--- |--- |--- |--- |--- |--- |--- |
| Effectively leveraging Multi-modal Features for Movie Genre Classification | [scholar](https://scholar.google.com/scholar?cluster=7914951466429935825&hl=en&as_sdt=0,5) | 2022 | embeddings + fusion + MLP | CLIP image embeddings, PANNs audio embeddings, CLIP text embeddings | MovieNet | movie genre classification |
| OS-MSL: One Stage Multimodal Sequential Link Framework for Scene Segmentation and Classification | [scholar](https://scholar.google.com/scholar?cluster=12395964150974750638&hl=en&as_sdt=0,5) | 2022 | embeddings + novel transformer | ResNet-18 image embeddings, ResNet-VLAD audio embeddings | TI-News | news scene segmentation/classification (studio, outdoor, interview) |
| Detection of Animated Scenes Among Movie Trailers | [scholar](https://scholar.google.com/scholar?cluster=6655397332509975816&hl=en&as_sdt=0,5) | 2022 | CNN + GRU | EfficientNet image embeddings | Private dataset | genre classification from movie trailer scenes |
| A multi-label movie genre classification scheme based on the movie's subtitles | [springer](https://link.springer.com/article/10.1007/s11042-022-12961-6) | 2022 | KNN | text frequency vectors | Private dataset | genre classification from movie subtitle text |
| A multimodal approach for multi-label movie genre classification | [scholar](https://scholar.google.com/scholar?cluster=9310008085751443571&hl=en&as_sdt=0,5) | 2020 | CNN + LSTM | MFCCs/SSD/LBP from audio, LBP/3DCNN from video frames, Inception-v3 from poster, TFIDF from text | Private dataset | genre classification from movie trailers |
| Genre classification of movie trailers using 3d convolutional neural networks | [ieee](https://ieeexplore.ieee.org/abstract/document/9121148) | 2020 | 3D CNN | images | Private dataset | genre classification from movie trailer scenes |
| A unified framework of deep networks for genre classification using movie trailer | [scholar](https://scholar.google.com/scholar?cluster=14452117618428848943&hl=en&as_sdt=0,5) | 2020 | CNN + LSTM | Inception V4 image embeddings | EmoGDB | genre classification from movie trailer scenes |
| Towards story-based classification of movie scenes | [scholar](https://scholar.google.com/scholar?cluster=3605673709859574181&hl=en&as_sdt=0,5) | 2020 | logistic regression | manually extracted categorical features | Flintstones Scene Dataset | scene classification (Obstacle, Midpoint, Climax of Act 1) |

### multimodal architectures

#### synchronous multimodal architectures

| name | paper | year | model | features | datasets | tasks | modalities |
|--- |--- |--- |--- |--- |--- |--- |--- |
| M&M Mix: A Multimodal Multiview Transformer Ensemble | [scholar](https://scholar.google.com/scholar?cluster=14299004100101928418&hl=en&as_sdt=0,5) | 2022 | transformer with 2 cls heads | ViT image embeddings from audio spect., frame image, optical flow | Epic-Kitchens | video/action classification | image + audio + optical flow |
| MultiMAE: Multi-modal Multi-task Masked Autoencoders | [scholar](https://scholar.google.com/scholar?cluster=7235983779434806126&hl=en&as_sdt=0,5) | 2022 | transformer with 3 decoder + cls heads | ViT-like image enc. patch embeddings (optional modalities) | ImageNet: Pseudo labeled multi-task training dataset (depth, segm) | image cs., semantic segm., depth est. | image + depth map |
| Data2vec: A general framework for self-supervised learning in speech, vision and language | [scholar](https://scholar.google.com/scholar?cluster=12686412422242429370&hl=en&as_sdt=0,5) | 2022 | single encoder | transformer based audio, text, image encoder embeddings | ImageNet, Librispeech | masked pretraining | image + audio + text |
| VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text | [scholar](https://scholar.google.com/scholar?cluster=7327595990658945420&hl=en&as_sdt=0,5) | 2022 | 1 encoder per modality | transformer based audio, text, image encoder embeddings | AudioSet, HowTo100M | pretraining + video/audio classification | image + audio + text |
| Expanding Language-Image Pretrained Models for General Video Recognition | [scholar](https://scholar.google.com/scholar?cluster=1144066736404687657&hl=en&as_sdt=0,5) | 2022 | 1 encoder per modality | transformer based video, text encoder embeddings | HMDB-51, UCF-101 | contrastive pretraining | video + text |
| Audio-Visual Instance Discrimination with Cross-Modal Agreement | [scholar](https://scholar.google.com/scholar?cluster=885326186401082715&hl=en&as_sdt=0,5) | 2021 | 1 encoder per modality | CNN based audio, video encoder embeddings | HMDB-51, UCF-101 | video/audio classification | video + audio |
| Robust Audio-Visual Instance Discrimination | [scholar](https://scholar.google.com/scholar?cluster=9200438422642254466&hl=en&as_sdt=0,5) | 2021 | 1 encoder per modality | CNN based audio, video encoder embeddings | HMDB-51, UCF-101 | video/audio classification | video + audio |
| Learning transferable visual models from natural language supervision | [scholar](https://scholar.google.com/scholar?cluster=15031020161691567042&hl=en&as_sdt=0,5) | 2021 | 1 encoder per modality | transformer based image, text encoder embeddings | JFT-300M | contrastive pretraining | image + text |
| Self-supervised multimodal versatile networks | [scholar](https://scholar.google.com/scholar?cluster=16748353423289036473&hl=en&as_sdt=0,5) | 2020 | multiple encoders | CNN based image/audio embeddings, word2vec text embeddings | UCF101, Kinetics, AudioSet | contrastive pretraining + classification | image + audio + text |
| Uniter: Universal image-text representation learning | [scholar](https://scholar.google.com/scholar?cluster=3224460637705754187&hl=en&as_sdt=0,5) | 2020 | multimodal encoder | combined embeddings | COCO, Visual Genome, Conceptual Captions | qa/image-text retrieval | image + text |
| 12-in-1: Multi-task vision and language representation learning | [scholar](https://scholar.google.com/scholar?cluster=17276757515931533114&hl=en&as_sdt=0,5) | 2020 | multimodal encoder | combined embeddings | COCO, Flickr30k | qa/image-text retrieval | image + text |
| Two-stream convolutional networks for action recognition in videos | [scholar](https://scholar.google.com/scholar?cluster=582514008712420788&hl=en&as_sdt=0,5) | 2014 | 1 encoder per modality | CNN based audio, text encoder embeddings | HMDB-51, UCF-101 | video/audio classification | video + optical flow |

#### asynchronous multimodal architectures

| name | paper | year | model | features | datasets | tasks | modalities |
|--- |--- |--- |--- |--- |--- |--- |--- |
| OmniMAE: Single Model Masked Pretraining on Images and Videos | [scholar](https://scholar.google.com/scholar?cluster=4636381983240187321&hl=en&as_sdt=0,5) | 2022 | transformer with 1 cls. head | ViT-like image/video enc. patch embeddings | ImageNet, SSv2 | video/action classification | image + video |
| OMNIVORE: A Single Model for Many Visual Modalities | [scholar](https://scholar.google.com/scholar?cluster=7775892631592466813&hl=en&as_sdt=0,5&as_ylo=2021) | 2022 | transformer with 3 cls. heads | ViT-like image/video enc. patch embeddings | ImageNet, Kinetics, SSv2, SUN RGB-D | image cls., action recog., depth est. | image + video + depth map |
| Polyvit: Co-training vision transformers on images, videos and audio | [scholar](https://scholar.google.com/scholar?cluster=2433441885724580400&hl=en&as_sdt=0,5) | 2021 | transformer with 9 cls. heads | ViT-like image/video/audio enc. embeddings | ImageNet, CIFAR, Kinetics, Moments in Time, AudioSet, VGGSound | image cls., video cls., audio cls. | image + video + audio |

### action recognition

#### with transformers

| name | paper | year | model | features | datasets | tasks |
|--- |--- |--- |--- |--- |--- |--- |
| Frozen CLIP Models are Efficient Video Learners | [scholar](https://scholar.google.com/scholar?cluster=16057670792750577500&hl=en&as_sdt=0,5) | 2022 | transformer with 1 cls head | CLIP image embeddings | ImageNet, Kinetics, SSv2 | action recognition |
| Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training | [scholar](https://scholar.google.com/scholar?cluster=8140812159859442226&hl=en&as_sdt=0,5) | 2022 | transformer with 1 cls head | ViT-like video enc. patch embeddings | Kinetics, SSv2 | action recognition |
| Bevt: Bert pretraining of video transformers | [scholar](https://scholar.google.com/scholar?cluster=9527303198700083047&hl=en&as_sdt=0,5) | 2022 | encoder-decoder transformer | VideoSwin image/video enc. embeddings | Kinetics, SSv2 | action recognition |
| Video swin transformer | [scholar](https://scholar.google.com/scholar?cluster=5833041667751260373&hl=en&as_sdt=0,5) | 2022 | Swin trans. with cls.head | Swin video enc. embeddings | Kinetics, SSv2 | action recognition |
| Is space-time attention all you need for video understanding? | [scholar](https://scholar.google.com/scholar?cluster=6828425192739736056&hl=en&as_sdt=0,5) | 2021 | transformer with cls. head | ViT-like video enc. patch embeddings | Kinetics, SSv2 | action recognition |

#### with 3D CNNs

| name | paper | year | model | features | datasets | tasks |
|--- |--- |--- |--- |--- |--- |--- |
| X3d: Expanding architectures for efficient video recognition | [scholar](https://scholar.google.com/scholar?cluster=5426206565542427464&hl=en&as_sdt=0,5) | 2020 | CNN with cls. head | 3D CNN based video enc. embeddings | Kinetics, SSv2 | action recognition |
| Slowfast networks for video recognition | [scholar](https://scholar.google.com/scholar?cluster=1892562522989461632&hl=en&as_sdt=0,5) | 2019 | CNN with cls. head | 3D CNN based video enc. embeddings | Kinetics, SSv2 | action recognition |
| A closer look at spatiotemporal convolutions for action recognition (R2+1D) | [scholar](https://scholar.google.com/scholar?cluster=9524036545693727210&hl=en&as_sdt=0,5) | 2018 | CNN with cls. head | 3D CNN based video enc. embeddings | Kinetics, HMDB-51, UCF-101 | action recognition |
| Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset (I3D) | [scholar](https://scholar.google.com/scholar?cluster=9581219496538221166&hl=en&as_sdt=0,5) | 2017 | CNN with cls. head | 3D CNN based video enc. embeddings | Kinetics, HMDB-51, UCF-101 | action recognition |

### contrastive representation learning

| name | paper | date |
|--- |--- |--- |
| Vatt: Transformers for multimodal self-supervised learning from raw video, audio and text | [scholar](https://scholar.google.com.tr/scholar?cluster=7327595990658945420) | 2021 |
| Supervised contrastive learning | [scholar](https://scholar.google.com.tr/scholar?cluster=2159203708190499110) | 2020 |

### review papers

| name | paper | date |
|--- |--- |--- |
| Machine Learning Models for Content Classification in Film Censorship and Rating | [pdf](https://www.researchgate.net/profile/Mahmudul-Haque-4/publication/360812725_Machine_Learning_Models_for_Content_Classification_in_Film_Censorship_and_Rating/links/629219a58d19206823e2cf70/Machine-Learning-Models-for-Content-Classification-in-Film-Censorship-and-Rating.pdf) | 2022 |
| A survey of artificial intelligence strategies for automatic detection of sexually explicit videos | [scholar](https://scholar.google.com/scholar?cluster=3622219384521520101&hl=en&as_sdt=0,5) | 2022 |
| A survey on video content rating: taxonomy, challenges and open issues | [pdf](https://link.springer.com/content/pdf/10.1007/s11042-021-10838-8.pdf) | 2021 |
| Multimodal Learning with Transformers: A Survey  | [scholar](https://scholar.google.com/scholar?cluster=6631830511436466415&hl=en&as_sdt=0,5) | 2022 |
| A Survey Paper on Movie Trailer Genre Detection | [scholar](https://scholar.google.com/scholar?cluster=18275675471389373237&hl=en&as_sdt=0,5) | 2020 |

## tools

| name | url | description |
|--- |--- |--- |
| safetext | [github](https://github.com/safevideo/safetext) | multilingual swear word detection and filtering from strings |
| PySceneDetect | [github](https://github.com/Breakthrough/PySceneDetect) | Python and OpenCV-based scene cut/transition detection program & library |
| LAION safety toolkit | [github](https://github.com/LAION-AI/LAION-SAFETY) | NSFW detector trained on LAION dataset |
| pysrt | [github](https://github.com/byroot/pysrt) | Python parser for SubRip (srt) files |
| ffsubsync | [github](https://github.com/smacke/ffsubsync) | Automagically synchronize subtitles with video. |
| MoviePy | [github](https://github.com/Zulko/moviepy) | Video editing with Python |
