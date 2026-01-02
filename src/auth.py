"""
User Authentication System
Handles user registration, login, and session management
"""
import sqlite3
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime, timedelta
from src.config import BASE_DIR

# Database path
DB_PATH = BASE_DIR / "users.db"


class UserAuth:
    """User authentication and management system"""
    
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize the user database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Create user sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_token TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256 with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, stored_hash = password_hash.split(':')
            computed_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return computed_hash == stored_hash
        except:
            return False
    
    def register_user(self, username: str, email: str, password: str, full_name: str = "") -> Tuple[bool, str]:
        """
        Register a new user
        
        Returns:
            (success: bool, message: str)
        """
        # Validation
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters long"
        
        if not email or '@' not in email:
            return False, "Please enter a valid email address"
        
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters long"
        
        # Check if user exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check username
        cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
        if cursor.fetchone():
            conn.close()
            return False, "Username already exists. Please choose another."
        
        # Check email
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            return False, "Email already registered. Please use another email."
        
        # Create user
        password_hash = self._hash_password(password)
        try:
            cursor.execute("""
                INSERT INTO users (username, email, password_hash, full_name)
                VALUES (?, ?, ?, ?)
            """, (username, email, password_hash, full_name))
            conn.commit()
            conn.close()
            return True, "Registration successful! You can now login."
        except Exception as e:
            conn.close()
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, username: str, password: str) -> Tuple[bool, Optional[dict], str]:
        """
        Authenticate user and create session
        
        Returns:
            (success: bool, user_data: dict or None, message: str)
        """
        if not username or not password:
            return False, None, "Please enter both username and password"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user
        cursor.execute("""
            SELECT id, username, email, password_hash, full_name, is_active
            FROM users WHERE username = ? OR email = ?
        """, (username, username))
        
        user = cursor.fetchone()
        
        if not user:
            conn.close()
            return False, None, "Invalid username or password"
        
        user_id, db_username, email, password_hash, full_name, is_active = user
        
        if not is_active:
            conn.close()
            return False, None, "Account is deactivated. Please contact administrator."
        
        # Verify password
        if not self._verify_password(password, password_hash):
            conn.close()
            return False, None, "Invalid username or password"
        
        # Update last login
        cursor.execute("""
            UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.now().isoformat(), user_id))
        
        # Create session token
        session_token = secrets.token_urlsafe(32)
        expires_at = (datetime.now() + timedelta(days=30)).isoformat()  # 30 days
        
        cursor.execute("""
            INSERT INTO user_sessions (user_id, session_token, expires_at)
            VALUES (?, ?, ?)
        """, (user_id, session_token, expires_at))
        
        conn.commit()
        conn.close()
        
        user_data = {
            'id': user_id,
            'username': db_username,
            'email': email,
            'full_name': full_name or db_username,
            'session_token': session_token
        }
        
        return True, user_data, "Login successful!"
    
    def verify_session(self, session_token: str) -> Optional[dict]:
        """Verify session token and return user data"""
        if not session_token:
            return None
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT u.id, u.username, u.email, u.full_name, s.expires_at
            FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.session_token = ? AND u.is_active = 1
        """, (session_token,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return None
        
        user_id, username, email, full_name, expires_at = result
        
        # Check if session expired
        if expires_at:
            try:
                expires_datetime = datetime.fromisoformat(expires_at)
                if expires_datetime < datetime.now():
                    return None
            except:
                return None
        
        return {
            'id': user_id,
            'username': username,
            'email': email,
            'full_name': full_name or username
        }
    
    def logout_user(self, session_token: str):
        """Remove session token"""
        if not session_token:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM user_sessions WHERE session_token = ?", (session_token,))
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_id: int) -> dict:
        """Get user statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT s.id) as session_count,
                MAX(s.created_at) as last_session
            FROM user_sessions s
            WHERE s.user_id = ?
        """, (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'session_count': result[0] or 0,
            'last_session': result[1] or 'Never'
        }

