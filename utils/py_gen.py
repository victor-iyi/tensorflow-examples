import os


def gen_data(file, max=50):
    """Generate Python dataset.

    Args:
        file (str):
            File to write the generated dataset into.

        max (int): default 50
            Maximum number of files to generate.
    """
    PYTHON_HOME = os.path.join('/Library/Frameworks/Python.framework',
                               'Versions/3.6/lib/python3.6/')
    if os.path.isfile(file):
        import shutil
        shutil.rmtree(os.path.dirname(file))

    if not os.path.isdir(os.path.dirname(file)):
        os.makedirs(os.path.dirname(file))

    handle = open(file, 'a', encoding='utf-8')

    counter = 0
    for (root, dirs, files) in os.walk(PYTHON_HOME):
        files = [os.path.join(root, f) for f in files if f.endswith('.py')]
        for _file in files:
            try:
                code = open(_file, 'r', encoding='utf-8').read()
                handle.write(str(code) + '\n')
            except Exception as e:
                print('Exception: {e}'.format(e))

        counter += 1
        if counter > max:
            break


if __name__ == '__main__':

    # Generate Python dataset.
    data_path = '../datasets/python_code.py'
    gen_data(data_path, max=50)
