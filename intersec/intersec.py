import os
import argparse
import subprocess


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input-file', required=True, help='File with users')
    parser.add_argument('--is-size', default=False, type=bool, help='If there is a degree of vertex at the beginning of every line.')
    parser.add_argument('--maxsize', default=600, type=int, help='Maximum sane degree of vertex in U-part. For example, user can\'t have more than *maxsize* subscriptions.')
    parser.add_argument('--nthreads', default=26, type=int, help='Number of threads to use. Currently maximum value is 675')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_file = args.input_file

    if not args.is_size:
        new_input_file = 'reformatted_input.txt'
        file1 = open(input_file)
        file2 = open(new_input_file, 'wt')

        cou = 0
        while True:
            res = file1.readline()
            if (res == '\n'):
                continue
            cou += 1

            if (len(res) == 0):
                break

            print(str(len(res.split())) + ' ' + res, file=file2)

        print(cou, flush=True)

        input_file = new_input_file

    res = os.system('g++ -std=c++11 intersec.cpp -lpthread -o intersec')
    proc = subprocess.Popen(('./intersec ' + input_file + ' ' + str(args.nthreads) + ' ' + str(args.maxsize)).split())
    print(proc.pid)
    proc.wait()
    res = os.system("cat cor_matrices/* > cor.out")
    res = os.system("cat jac_matrices/* > jac.out")


main()