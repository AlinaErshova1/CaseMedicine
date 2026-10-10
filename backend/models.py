from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey, Float, Text, Date
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from database import Base


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String, nullable=False)
    login = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    specialty = Column(String, nullable=True)   # «невролог», «рентгенолог»
    hospital = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Пациенты, которых создал этот врач
    patients = relationship(
        "Patient",
        foreign_keys="Patient.created_by",
        back_populates="created_by_doctor"
    )
    # Пациенты, которых этот врач лечит
    attending_patients = relationship(
        "Patient",
        foreign_keys="Patient.attending_doctor_id",
        back_populates="attending_doctor"
    )


class Admin(Base):
    __tablename__ = "admins"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    full_name = Column(String, nullable=False)
    login = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Patient(Base):
    __tablename__ = "patients"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    anonymized_code = Column(String, unique=True, nullable=False, index=True)

    # === Персональные данные (шифруем) ===
    full_name_encrypted = Column(Text, nullable=False)
    phone_encrypted = Column(Text, nullable=True)
    email_encrypted = Column(Text, nullable=True)
    address_encrypted = Column(Text, nullable=True)
    policy_oms_encrypted = Column(Text, nullable=True)  # полис ОМС — тоже ПДн

    # === Демография (не шифруем) ===
    birth_date = Column(Date, nullable=True)
    sex = Column(String, nullable=True)   # "M" / "F"

    # === Медицинская информация ===
    blood_type = Column(String, nullable=True)      # "II (A) Rh+"
    height_cm = Column(Integer, nullable=True)
    weight_kg = Column(Integer, nullable=True)
    reason = Column(Text, nullable=True)            # Причина обращения
    complaints = Column(Text, nullable=True)        # Жалобы
    allergies = Column(Text, nullable=True)         # Аллергии
    notes = Column(Text, nullable=True)

    # === Связи ===
    created_by = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)
    attending_doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # === Relationships ===
    created_by_doctor = relationship(
        "Doctor", foreign_keys=[created_by], back_populates="patients"
    )
    attending_doctor = relationship(
        "Doctor", foreign_keys=[attending_doctor_id], back_populates="attending_patients"
    )
    studies = relationship(
        "Study", back_populates="patient",
        cascade="all, delete-orphan",
        order_by="Study.created_at.desc()"
    )


class Study(Base):
    __tablename__ = "studies"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False)

    original_filename = Column(String)
    original_path = Column(String)
    result_path = Column(String, nullable=True)

    prediction = Column(String, nullable=True)   # glioma / meningioma / pituitary / no_tumor
    confidence = Column(Float, nullable=True)
    status = Column(String, default="pending")   # pending / done / error

    created_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)

    patient = relationship("Patient", back_populates="studies")