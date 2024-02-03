import requests
import subprocess
import os

def download_file(url, destination):
    response = requests.get(url, stream=True)
    with open(destination, 'wb') as file:
        for chunk in response.iter_content(chunk_size=128):
            file.write(chunk)

def main():
    file_url = 'https://github.com/marczwalua/systemc/raw/master/systemc-2.3.3.tar.gz'
    file_destination = 'systemc-2.3.3.tar.gz'

    if os.path.exists(file_destination):
        print(f"The file '{file_destination}' exists.")
    else:
        print(f"The file '{file_destination}' does not exist.")
        print(f'Downloading {file_url}...')
        download_file(file_url, file_destination)
        print('Download complete.')

    print('Building Docker image...')
    subprocess.run(['docker', 'build', '-t', 'systemc', '.'])
    print('Docker build complete.')

if __name__ == "__main__":
    main()