from sentimentalize import create_app
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from users.models import User

app = create_app()
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

actions = "{1} - Database create_all\n" \
          "{2} - Create new user\n" \
          "{0} - Exit" \

action = 5
while action != 0:
    print(actions)
    print("Please choose an option.")
    try:
        action = int(input('> '))
    except ValueError:
        print('Invalid option')

    if action == 2:
        confirmation = str.lower(input('Have you run the {1} option yet? (y/n) '))
        if confirmation not in ['y', 'n']:
            print("Invalid option")
            continue
        if confirmation == "n":
            continue
        username = str(input('Username: '))
        email = str(input('Email: '))
        password = str(input('Password: '))

        with app.app_context():
            hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_pw)

        with app.app_context():
            db.session.add(user)
            db.session.commit()

        print('User created successfully')

    if action == 1:
        with app.app_context():
            db.create_all()
        print("The database was created successfully!")

