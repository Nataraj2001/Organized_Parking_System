from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class ParkingSpace(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(20), nullable=False, default='empty')
    coordinates = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return f'<ParkingSpace {self.id} {self.status}>'
