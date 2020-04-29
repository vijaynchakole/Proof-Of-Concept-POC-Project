### Project Name : Directory Operation 

Automation script which display all files , find checkSum of files, display duplicate files, remove duplicate files

1). Directory Watcher

2). Directory File CheckSum

3). Directory Duplicate File Detector

4). Directory duplicate File Removal

### Dependencies :

os : file system related operations

sys : commandline argument

hashlib : use for finding checksum


#### Project File Name : Project_file_Directory_Operations.py

Code Explanation :

Function Name and their Working :

1). **Function Name** : *DirectoryWatcher( )*

input : folder name

Working :  iterate through the folder and find and display subfolder and file names within that folder

2). **Function Name** : *hashfile( )* 

input : folder name

working : find checksum and return hexadigit


3). **Function Name** :  *hashfile( )*

input : folder name

working : display checksum of every file within in that folder

4). **Function Name** : *FindDuplicate( )*

input : folder name

working : find duplicate files


5). **Function Name** : *PrintDuplicate( )*

input : folder name

working : print duplicate file names


6). **Function Name** : *DeleteFiles( )*

input : dictionary which contains file name as value and their checksum as key 

working : delete duplicate files within given folder


7). **Function Name** : *PrintResults ( )*

input : dictionary which contains file name as value and their checksum as key 

working : display duplicate files


8). **Function Name** : *main( )*

working : calling above mentioned function

Note : entry point function


