import autocode
import argparse

argp = argparse.ArgumentParser()
argp.add_argument("--share", help="share workspace online", action="store_true")
argp.add_argument("--auth", help="require auth to access workspace (split credentials with \":\")", type=str, default="", action="store")
args = argp.parse_args()

if len(args.auth) > 0:
    auth = args.auth.split(":")
else:
    auth = None

autocode.webapp(args.share, auth)