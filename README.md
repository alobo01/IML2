### Team

Antonio Lobo Santos
Pedro Agundez Fernandez


### Running script for the first time
These sections show how to create virtual environment for
our script and how to install dependencies
1. Open folder in terminal
```bash
cd <root_folder_of_project>/
```
2. Create virtual env
```bash
python3 -m venv venv/
```
3. Open virtual env
```bash
source venv/bin/activate
```
4. Install required dependencies
```bash
pip install -r requirements.txt
```
you can check if dependencies were installed by running next
command,it should print list with installed dependencies
```bash
pip list
```
5. Close virtual env
```bash
deactivate
```
## Execute scripts
1.open virtual env
```bash
source venv/bin/activate
```
2. Running the script (Run it without arguments to display help menu)
 ```bash
 python3 main.py
 ```

3. Close virtual env
```bash
deactivate
```