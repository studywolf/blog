============================================
Studywolf code blog
============================================

This repository serves to hold projects that I'm both working on 
and have completed and presented on http://www.studywolf.com.

Control repo
============

Installation
------------

The control directory requires that you have docopt installed::

   pip install docopt
   
Additionally, there are a number of arm models available, if you 
wish to use anything other than the 2 link arm coded in python, 
then you will have to compile the arm. You can compile the arms by
going in to the Arms/num_link/ folder and running setup::

   python setup.py build_ext -i
   
This will compile the arm for you into a shared object library that's
accessible from Python. 

NOTE: This has only been tested on linux and currently does not compile on Mac. 

Running
-------

To run the basic control code, from the base directory::

   python run.py ARM CONTROL TASK
   
Where ARM = (arm1 | arm2 | arm2_python | arm3), the control types 
available are CONTROL = (dmp | gc | trajectory | osc), and the tasks
are those listed in the task directory, examples include 
TASK = (follow | random | walk | write_numbers | write_words).
