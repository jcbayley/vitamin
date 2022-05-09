======
Create some gw data
======

To generate some data without using the make condor_files script you can use the generate_data module

Generate training data with 

.. code-block:: 

   $python -m vitamin.gw.generate_data --start-ind 0 --num-files 20 --run-type training --ini-file config.ini

if you want to generate further files after running this once then change the start-ind to the next available file number (in this case it sould be 20 as 20 files are produced using this script) if this value is not changed then files will be overwritten.
To create all files at once just change the --num-files parameter to the number of files that you would like. 

To genereate validation data run the script

.. code-block:: 

   $python -m vitamin.gw.generate_data --start-ind 0 --num-files 1 --run-type validation --ini-file config.ini


To generate test data use the script below, in this case the sampler of choice is nessai, the only other choice currently is dynesty.
To generate another peice of test data/samples change the --start-ind to another value, otherwise previous runs will be overwritten.


.. code-block:: 

   $python -m vitamin.gw.generate_data --start-ind 0 --num-files 1 --run-type test --ini-file config.ini --sampler "nessai"


