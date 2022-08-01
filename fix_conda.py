import os, argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    args = parser.parse_args()
    if args.env:
        env_name = args.env
    else:
        exit()
    env_path=f"/work02/home/wlluo/.conda/envs/{env_name}"
    pythons = os.listdir(os.path.join(env_path, "lib"))
    for python in pythons:
        s = python.split(".")
        if s[0] == "python3" and len(s) == 2:
            site = os.path.join(env_path, "lib", python, "site-packages")
            print(site)
            with open(os.path.join(env_path, "lib", python, "site.py")) as f:
                lines = f.readlines()
            
            newlines = []
            for line in lines:
                if line[:12] == "USER_SITE = ":
                    print(f"{line} to USER_SITE = {site}\n")
                    newlines.append(f"USER_SITE = \"{site}\"\n")
                elif line[:12] == "USER_BASE = ":
                    print(f"{line} to USER_BASE = {env_path}\n")
                    newlines.append(f"USER_BASE = \"{env_path}\"\n")
                else:
                    newlines.append(line)
            with open(os.path.join(env_path, "lib", python, "site.py"), 'w+') as f2:
                for newline in newlines:
                    f2.write(newline)
    # cd /work02/home/wlluo/.conda/envs/$env_name/lib/python3.*
    # site=`pwd`/site-packages
    # echo $env_name
    # sed -n "/USER_SITE = None/p" site.py
    # sed -n "/USER_BASE = None/p" site.py
    # #sed -n "/USER_SITE = None/USER_SITE = ${site}/p" site.py
    # #sed -n "/USER_BASE = None/USER_SITE = ${env_path}/p" site.py