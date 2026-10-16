from sqlalchemy.orm import Session
from models import Doctor, Admin
from security import hash_password, verify_password
from logger import write_log


def create_doctor(db: Session, full_name: str, login: str, password: str,
                  specialty: str = None, hospital: str = None, admin_id: str = None):
    doctor = Doctor(
        full_name=full_name,
        login=login,
        password_hash=hash_password(password),
        specialty=specialty,
        hospital=hospital
    )
    db.add(doctor)
    db.commit()
    db.refresh(doctor)

    write_log("create_doctor", admin_id, f"doctor_id={doctor.id}")
    return doctor


def create_admin(db: Session, full_name: str, login: str, password: str, admin_id: str = None):
    admin = Admin(
        full_name=full_name,
        login=login,
        password_hash=hash_password(password)
    )
    db.add(admin)
    db.commit()
    db.refresh(admin)

    write_log("create_admin", admin_id, f"new_admin_id={admin.id}")
    return admin


def get_all_doctors(db: Session):
    return db.query(Doctor).all()


def get_all_admins(db: Session):
    return db.query(Admin).all()