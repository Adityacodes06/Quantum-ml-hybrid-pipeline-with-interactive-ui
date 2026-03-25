from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database configuration
DATABASE_URI = 'postgresql://username:password@hostname:port/database'

# Create engine
engine = create_engine(DATABASE_URI)

# Session management
Session = sessionmaker(bind=engine)

# Base class for user models
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(100), nullable=False)

    def __repr__(self):
        return f'<User {self.username}>'