### Background and Tools
Just some links to resources / key words to look up, that are often useful getting started. 

Use as you see fit / as it comes up. 

#### Linux introduction 

https://ryanstutorials.net/linuxtutorial/

(in particular the command-line stuff)

#### Git introduction 
http://rogerdudler.github.io/git-guide/ 

starting from 'git clone' and with https://git.hhu.de/alden101/utr-grappling.git as /path/to/repository 

#### HPC introduction 

Get access here: https://www.zim.hhu.de/forschung/high-performance-computing under "HPC Antrag"
Project exists "HelixerOpt", I am the project lead

the hhu wiki page is the system-specific reference here: https://wiki.hhu.de/display/HPC/Entwicklungs-Server 

also, googling PBS/qsub/etc... 

you will need to ssh to the cluster, and be able to module avail, load modules, qsub, qstat, qdel 

it also might be good to try an interactive session https://wiki.hhu.de/display/HPC/Entwicklungs-Server 

#### Predictive modelling intro 
You might take a look at at least: https://www.youtube.com/watch?v=MyBSkmUeIEs 

For a more thorough look, I'd encourage you to work through: https://www.coursera.org/learn/machine-learning (make sure to find the 'Enroll for Free' button) whenever you have down time during your work here. 

#### ssh keys
To access our network you have to have an ssh-key (password alone is blocked). 

Some make-it-work info on making ssh keys (obviously there are more thorough explanations out there if you look for them): 
https://cloud.denbi.de/wiki/quickstart/#ssh-setup

Once you do this on your home computer, I will need the public key (.pub), you can send it to me (Ali) on teams, via email or similar.

Finally, if you haven't been using ssh-keys before I personally recommend dropping the `-f new_id` part from the linked instructions, and just save it to the default location. It normally makes ones life a bit easier later. 

Also if you have used ssh-keys before, already have an ssh-key (and if required also know the password for said key), then please just send me your existing public key.

