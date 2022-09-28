# deep-learning-content-moderation
Various sources for deep learning based content moderation, sensitive content detection, scene genre classification from text, audio, video &amp; image input modalities.

## datasets

| name | paper | year | url | input modality | task | labels |
|--- |--- |--- |--- |--- |--- |--- |
| LSPD | [pdf](https://quanghuy0497.github.io/pdf/LSPD_IJIES.pdf) | 2022 | [page](https://sites.google.com/uit.edu.vn/LSPD) | image | image classification, instance segmentation | porn, normal, sexy, hentai, drawings, female/male genital, female breast, anus |
| Movie script dataset | [pdf](https://ojs.aaai.org/index.php/AAAI/article/view/3844/3722) | 2019 | [github]([pdf](https://ojs.aaai.org/index.php/AAAI/article/view/3844/3722)) | text | text classification | violent or not |
| Nudenet | [github](https://github.com/notAI-tech/NudeNet) | 2019 | [archive.org](https://archive.org/details/NudeNet_classifier_dataset_v1) | image | image classification | nude or not |
| Adult content dataset | [pdf](https://arxiv.org/abs/1612.09506) | 2017 | [contact](https://drive.google.com/file/d/1MrzzFxQ9t56i9d3gMAI3SaHXfVd1UUDt/view) | image | image classification | nude or not |
| Substance use dataset | [pdf](https://web.cs.ucdavis.edu/~hpirsiav/papers/substance_ictai17.pdf) | 2017 | [first author](https://www.linkedin.com/in/roy-arpita/) | image | image classification | drug related or not |
| Pornography2k | [pdf](https://www.researchgate.net/profile/Sandra-Avila-5/publication/308398120_Pornography_Classification_The_Hidden_Clues_in_Video_Space-Time/links/5a205361a6fdcccd30e00d4a/Pornography-Classification-The-Hidden-Clues-in-Video-Space-Time.pdf) | 2016 | [contact](https://www.ic.unicamp.br/~rocha/mail/index.html) | video | video classification | porn or not |
| Violent Scenes Dataset | [springer](https://link.springer.com/article/10.1007/s11042-014-1984-4) | 2014 | [page](https://www.interdigital.com/data_sets/violent-scenes-dataset) | video | video classification | blood, fire, gun, gore, fight |
| VSD2014 | [pdf](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_cbmi_2015.pdf) | 2014 | [download](http://www.cp.jku.at/datasets/VSD2014/VSD2014.zip) | video | video classification | blood, fire, gun, gore, fight |
| AIIA-PID4 | [pdf](https://www.igi-global.com/gateway/article/full-text-pdf/79140&riu=true) | 2013 | - | image | image classification | bikini, porn, skin, non-skin |
| NDPI video dataset | [pdf](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.714.5143&rep=rep1&type=pdf) | 2013 | [page](https://sites.google.com/site/pornographydatabase/) | video | video classification | porn or not |
| HMDB-51 |--- | 2011 | [page](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#overview) | video |video classification | smoke, drink |




## papers

### sensitive content detection

| name | paper | year | model | features | datasets | tasks |
|--- |--- |--- |--- |--- |--- |--- |
| --- | --- | --- | --- | --- | --- |
| --- | --- | --- | --- | --- | --- |

### movie scene classification

| name | paper | year | model | features | datasets | tasks |
|--- |--- |--- |--- |--- |--- |--- |
| Effectively leveraging Multi-modal Features for Movie Genre Classification | [scholar](https://scholar.google.com/scholar?cluster=7914951466429935825&hl=en&as_sdt=0,5) | 2022 | embeddings + fusion + MLP | CLIP image embeddings, PANNs audio embeddings, CLIP text embeddings | MovieNet | movie genre classification |
| OS-MSL: One Stage Multimodal Sequential Link Framework for Scene Segmentation and Classification | [scholar](https://scholar.google.com/scholar?cluster=12395964150974750638&hl=en&as_sdt=0,5) | 2022 | embeddings + novel transformer | ResNet-18 image embeddings, ResNet-VLAD audio embeddings | TI-News | news scene segmentation/classification (studio, outdoor, interview) |
| Detection of Animated Scenes Among Movie Trailers | [scholar](https://scholar.google.com/scholar?cluster=6655397332509975816&hl=en&as_sdt=0,5) | 2022 | CNN + GRU | EfficientNet image embeddings | Private dataset | genre classification from movie trailer scenes |
| Genre classification of movie trailers using 3d convolutional neural networks | [scholar](https://scholar.google.com/scholar?cluster=14452117618428848943&hl=en&as_sdt=0,5) | 2020 | 3D CNN | images | Private dataset | genre classification from movie trailer scenes |
| A unified framework of deep networks for genre classification using movie trailer | [scholar](https://scholar.google.com/scholar?cluster=14452117618428848943&hl=en&as_sdt=0,5) | 2020 | CNN + LSTM | Inception V4 image embeddings | EmoGDB | genre classification from movie trailer scenes |
| Towards story-based classification of movie scenes | [scholar](https://scholar.google.com/scholar?cluster=3605673709859574181&hl=en&as_sdt=0,5) | 2020 | logistic regression | manually extracted categorical features | Flintstones Scene Dataset | scene classification (Obstacle, Midpoint, Climax of Act 1) |

### multimodal classification

| name | paper | year | model | features | datasets | tasks |
|--- |--- |--- |--- |--- |--- |--- |
| OMNIVORE: A Single Model for Many Visual Modalities | [scholar](https://scholar.google.com/scholar?cluster=7775892631592466813&hl=en&as_sdt=0,5&as_ylo=2021) | 2022 | transformer with 3 cls heads | ViT-like image/video enc. patch embeddings | ImageNet, Kinetics, SSv2, SUN RGB-D | (mt) image cls., action recog., depth est. |
| Frozen CLIP Models are Efficient Video Learners | [scholar](https://scholar.google.com/scholar?cluster=16057670792750577500&hl=en&as_sdt=0,5) | 2022 | transformer with 1 cls head | CLIP image embeddings | ImageNet, Kinetics, SSv2 | action recognition |
| M&M Mix: A Multimodal Multiview Transformer Ensemble | [scholar](https://scholar.google.com/scholar?cluster=14299004100101928418&hl=en&as_sdt=0,5) | 2022 | transformer with 2 cls heads | ViT image embeddings from audio spect., frame image, optical flow | Epic-Kitchens | (mt) video/action classification |
| OmniMAE: Single Model Masked Pretraining on Images and Videos | [scholar](https://scholar.google.com/scholar?cluster=4636381983240187321&hl=en&as_sdt=0,5) | 2022 | transformer with 1 cls head | ViT-like image/video enc. patch embeddings | ImageNet, SSv2 | video/action classification |
| MultiMAE: Multi-modal Multi-task Masked Autoencoders | [scholar](https://scholar.google.com/scholar?cluster=7235983779434806126&hl=en&as_sdt=0,5) | 2022 | transformer with 3 decoder+ cls heads | ViT-like image enc. patch embeddings (optional modalities) | ImageNet: Pseudo labeled multi-task training dataset (depth, segm) | (mt) image cs., semantic segm., depth est. |
| Data2vec: A general framework for self-supervised learning in speech, vision and language | [scholar](https://scholar.google.com/scholar?cluster=12686412422242429370&hl=en&as_sdt=0,5) | 2022 | single encoder | transformer based audio, text, image encoders | ImageNet, Librispeech | masked pretraining |

### review papers

| name | paper | date |
|--- |--- |--- |
| Machine Learning Models for Content Classification in Film Censorship and Rating |[pdf](https://www.researchgate.net/profile/Mahmudul-Haque-4/publication/360812725_Machine_Learning_Models_for_Content_Classification_in_Film_Censorship_and_Rating/links/629219a58d19206823e2cf70/Machine-Learning-Models-for-Content-Classification-in-Film-Censorship-and-Rating.pdf) | 2022 |
| A survey on video content rating: taxonomy, challenges and open issues |[pdf](https://link.springer.com/content/pdf/10.1007/s11042-021-10838-8.pdf) | 2021 |
| Multimodal Learning with Transformers: A Survey  |[scholar](https://scholar.google.com/scholar?cluster=6631830511436466415&hl=en&as_sdt=0,5) | 2022 |
| A Survey Paper on Movie Trailer Genre Detection |[scholar](https://scholar.google.com/scholar?cluster=18275675471389373237&hl=en&as_sdt=0,5) | 2020 |