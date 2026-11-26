"""Account management from the server's own terminal (e.g. a cloud machine without a local browser).

The web setup endpoint only accepts the machine itself, so on a remote server the first admin
is created here:

    python -m src.auth.cli create-admin <username>      # asks for the password twice
    python -m src.auth.cli reset-password <username>    # also signs out every device
    python -m src.auth.cli list
"""

import argparse
import getpass
import sys

from src.auth.users import ROLE_ADMIN, create_user, get_user, hash_password, list_users, revoke_tokens

MIN_PASSWORD_LENGTH = 8


def _ask_password() -> str:
    password = getpass.getpass("Mật khẩu: ")
    if len(password) < MIN_PASSWORD_LENGTH:
        sys.exit(f"Mật khẩu phải có ít nhất {MIN_PASSWORD_LENGTH} ký tự.")
    if getpass.getpass("Nhập lại mật khẩu: ") != password:
        sys.exit("Hai lần nhập mật khẩu không khớp.")
    return password


def _set_password(username: str, password: str) -> None:
    from src.auth import users

    data = users._load_users()
    key = username.strip().lower()
    data[key]["password_hash"] = hash_password(password)
    users._save_users(data)
    revoke_tokens(key)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.auth.cli", description="Quản lý tài khoản đăng nhập")
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create-admin", help="Tạo tài khoản quản trị")
    create.add_argument("username")
    reset = sub.add_parser("reset-password", help="Đặt lại mật khẩu (đăng xuất mọi thiết bị)")
    reset.add_argument("username")
    sub.add_parser("list", help="Liệt kê tài khoản")
    args = parser.parse_args(argv)

    if args.command == "create-admin":
        if len(args.username.strip()) < 3:
            sys.exit("Tên đăng nhập phải có ít nhất 3 ký tự.")
        try:
            user = create_user(args.username, _ask_password(), role=ROLE_ADMIN)
        except ValueError as e:
            sys.exit(str(e))
        print(f"Đã tạo tài khoản quản trị '{user.username}'.")
    elif args.command == "reset-password":
        if get_user(args.username) is None:
            sys.exit(f"Không có tài khoản '{args.username}'.")
        _set_password(args.username, _ask_password())
        print("Đã đặt lại mật khẩu; mọi thiết bị đã bị đăng xuất.")
    else:
        for user in list_users():
            print(f"{user['username']}\t{user['role']}")


if __name__ == "__main__":
    main()
