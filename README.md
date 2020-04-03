Copyright (C) 2020 by RWTH Aachen University                      
http://www.rwth-aachen.de                                             
                                                                         
License:                                                                                                                                       
This software is dual-licensed under:                                 
• Commercial license (please contact: lfb@lfb.rwth-aachen.de)         
• AGPL (GNU Affero General Public License) open source license        

# MLL

Welcome to the Machine Learning Laboratory offered by the Institute of Imaging and Computer Vision at RWTH Aachen University.
The course is offered every semester and covers 9 Sessions in total.
It is designed to start at the basics of Machine Learning, and then progresses until the SoTA of Deep Learning is reached.
After the course, you will have a understanding of both the theoretical concepts of ML as well as how they transition to code.
As we are a Computer Vision Institute, we will cover mostly ML in Computer Vision, though the basic concepts can be easily applied to other fields.
Inside the PreparationSheets folder, you can find preparatory material for each Session designed to give you the theoretical background
After having worked through the preparatory materials, you will then practice the theory with a coding exercise using [jupyter notebooks](https://jupyter.org/).

## How to run the coding exercises locally?

1. Install [docker](https://www.docker.com/)
2. Pull the docker container accompanying the Machine Learning Laboratory. You can do this using `docker pull lfbbot/mll` inside a terminal.
3. Start the docker container using `docker run -p 8000:8000 -d -v $YOUR_DIRECTORY:/home --name $NAME --rm lfbbot/mll`. `$YOUR_DIRECTORY` Is the folder on your local filesystem you want to save your progress to. This git repository will be cloned there. `$NAME` will be used to shut down your docker container later on.
4. A jupyterhub will be available at `localhost:8000`. Login using `praktikum` for both username and password. Your progress will be saved in `$YOUR_DIRECTORY`
5. After you are done, you can stop your docker container using `docker kill $NAME`.

* Note: The above has been tested using Ubuntu 18.04 only, though docker is also available on Windows and MacOS

## FAQ

1. Is it possible to get the ETCS for the course without having gained a slot via RWTHonline?
    * **No**, it is not, as we don't have an automated grading scheme available.
2. Sessions utilizing Deep-Learning take long on my local hardware. Is there a way to access better one?
    * While we cannot grant you access to high performance computing hardware, we refer you to Google Colab, as:
        * You can load public notebooks directly into [Google CoLab](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=WzIRIt9d2huC)
        * Missing dependencies can also be [installed](https://colab.research.google.com/notebooks/snippets/importing_libraries.ipynb#scrollTo=GQ18Kd5F3uKe)
    * If you have sufficient GPU-acceleration, i.e. NVIDIA with 6 GB of Graphics RAM, you can also [install nvidia-docker](https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0)) to make use of that.

## Funding
The Machine Learning Laboratory was supported by the German Federal Ministry of Education and Research (BMBF, Funding number: 01IS17072)

![BMBF_Logo](./BMBF_Logo.svg)
