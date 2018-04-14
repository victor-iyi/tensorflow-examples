import os

python_home = os.path.join('/Library/Frameworks/Python.framework',
                           'Versions/3.6/lib/python3.6/')


def main():
    handle = open('datasets/python_code.py', 'a', encoding='utf-8')
    max_loop = 50

    counter = 0
    for (root, dirs, files) in os.walk(python_home):
        files = [os.path.join(root, f) for f in files if f.endswith('.py')]
        for file in files:
            try:
                code = open(file, 'r', encoding='utf-8').read()
                handle.write(str(code) + '\n')
            except Exception as e:
                print('Exception: {e}'.format(e))

        counter += 1
        if counter > max_loop:
            break


if __name__ == '__main__':
    main()
