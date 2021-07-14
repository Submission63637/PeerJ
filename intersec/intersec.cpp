#include <map>
#include <unordered_map>
#include <vector>
#include <set>
#include <iostream>
#include <fstream>
#include <time.h>
#include <mutex>
#include <thread>
#include <cmath>

using namespace std;

signed main(int argc, char** argv)
{
    unordered_map <int, vector<int>> users_reversed;

    const int n_threads = stoi(argv[2]);
    const int MAXSIZE = stoi(argv[3]);
    vector <vector<int>> users;

    ifstream inp_adj(argv[1], ios_base::in);

    int v_deg = 0;
    int v_id = -1;

    while (inp_adj >> v_deg)
    {

        users.resize(users.size() + 1);
        v_id++;

        for (int i = 0; i < v_deg; i++)
        {
            int u_id;
            inp_adj >> u_id;

            users[v_id].push_back(u_id);
        }
    }

    int tttime = time(0);
    for (int i = 0; i < users.size(); i++)
    {
        for (int j = 0; j < users[i].size(); j++)
        {
            users_reversed[users[i][j]].push_back(i);
        }
    }

    int cou = 0;
    int ttt = time(0);

    const unsigned num_threads = n_threads;
    mutex iomutex;

    vector<thread> threads(num_threads);
    vector<int> uncounted_users(users.size());

    for (int i = 0; i < users.size(); i++)
    {
        for (int j = 0; j < users[i].size(); j++)
        {
            if (users_reversed.at(users[i][j]).size() >= MAXSIZE)
            {
                uncounted_users[i] += 1;
            }
        }
    }
    cout << users.size() << endl;
    for (unsigned num_t = 0; num_t < num_threads; num_t++)
    {
        threads[num_t] = thread([&iomutex, num_t, &users, &users_reversed, &uncounted_users, MAXSIZE, num_threads]
        {
            string out_dir_cor = "cor_matrices";
            string out_dir_jak = "jac_matrices";

            vector<int> matrix(users.size());
            string ind_t = "";
            ind_t += ('A' + num_t / 26);
            ind_t += ('A' + num_t % 26);
            string filename_c = out_dir_cor + "/cor_" + ind_t + ".out";
            string filename_j = out_dir_jak + "/jac_" + ind_t + ".out";
            ofstream onp_c(filename_c, ios_base::out | ios_base::binary);
            ofstream onp_j(filename_j, ios_base::out | ios_base::binary);

            for (int i = (users.size() / num_threads + 1) * num_t; i < min(users.size(), (users.size() / num_threads + 1) * (num_t + 1)); i++)
            {
                for (int k = 0; k < users.size(); k++)
                {
                    matrix[k] = 0;
                }

                for (int j = 0; j < users[i].size(); j++)
                {
                    if (users_reversed.at(users[i][j]).size() >= MAXSIZE)
                    {
                        continue;
                    }
                    for (auto k = users_reversed.at(users[i][j]).begin(); k != users_reversed.at(users[i][j]).end(); k++)
                    {
                        matrix[*k] += 1;
                    }
                }

                for (int k = 0; k < users.size(); k++)
                {
                    float cov;
                    float jac;
                    jac = matrix[k];

                    long long a = users[i].size() - uncounted_users[i];
                    long long b = users[k].size() - uncounted_users[k];

                    jac /= (a + b - jac);
                    cov = matrix[k] * users_reversed.size();
                    cov -= (a * b);
                    cov /= sqrt((float)a * b * (users_reversed.size() - a) * (users_reversed.size() - b));

                    onp_j.write((char *)&jac, sizeof(jac));
                    onp_c.write((char *)&cov, sizeof(cov));
                }
            }
            onp_c.close();
            onp_j.close();
        });
    }

    for (auto& t : threads)
    {
        t.join();
    }

    return 0;
}