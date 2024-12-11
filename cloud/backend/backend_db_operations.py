import aioodbc
import datetime, json
from pydantic import BaseModel, EmailStr
from base_logger import log_function

# Data Transfer Object (DTO) for user updates
class UserUpdateDTO(BaseModel):
    username: str
    email: EmailStr  # Validated email format
    role: str
    last_login: datetime.date
    created_at: datetime.datetime
    password_hash: str

# TODO: Move the database to a cloud server for better scalability and availability.

# Database connection configuration string
connection_string = "DRIVER={SQL Server};SERVER=DESKTOP-8UQUVJ2;DATABASE=capstonedesign_DB;Trust_Connection=yes"

@log_function
async def get_connection():
    """
    Initializes and returns an async database connection using aioodbc.
    """
    return await aioodbc.connect(dsn=connection_string, autocommit=True)

# Can be used for adding any data
@log_function
async def add_data(table, data):
    """
    Inserts a new user record into the specified table.

    Args:
        table (str): The table name to insert into.
        data (dict): A dictionary of column names and values.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?'] * len(data))
            sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            values = tuple(data.values())
            print("Values to insert:", values)
            await cursor.execute(sql, values)
            await conn.commit()
            print("Record added successfully.")

@log_function
async def delete_user(table, username):
    """
    Deactivates a user by setting their role to 'passive'.

    Args:
        table (str): The table name.
        username (str): The username of the user to deactivate.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            sql = f"UPDATE {table} SET role='passive' WHERE username=?"
            await cursor.execute(sql, (username,))
            await conn.commit()
            print("User deleted successfully.")

@log_function
async def show_all(table):
    """
    Fetches and returns all records from the specified table.

    Args:
        table (str): The table name.

    Returns:
        list: A list of rows from the table.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            sql = f'SELECT * FROM {table}'
            await cursor.execute(sql)
            rows = await cursor.fetchall()
            return rows

@log_function
async def get_all_visitor_logs_with_usernames():
    """
    Fetches visitor logs along with associated usernames by joining tables.

    Returns:
        list: A list of dictionaries containing log details and usernames.
    """
    sql = """
    SELECT 
        Visitor_Logs.log_id,
        Visitor_Logs.timestamp,
        Visitor_Logs.recognized,
        Users.username
    FROM 
        Visitor_Logs
    INNER JOIN Faces ON Visitor_Logs.face_id = Faces.face_id
    INNER JOIN Users ON Faces.user_id = Users.user_id;
    """
    try:
        async with await get_connection() as conn:  
            async with conn.cursor() as cursor:
                await cursor.execute(sql)
                rows = await cursor.fetchall()
                result = [
                    {
                        "log_id": row[0],
                        "timestamp": row[1],
                        "recognized": row[2],
                        "username": row[3]
                    }
                    for row in rows
                ]
                return result
    except Exception as e:
        return None
async def get_custom_data_from_custom_table(table: str, columns: list, conditions: dict = None):
    column_list = ', '.join(columns)
    sql = f"SELECT {column_list} FROM {table}"
    values = ()
    if conditions is not None:
        where_clause = ' AND '.join([f"{key} = ?" for key in conditions.keys()])
        sql += f" WHERE {where_clause}"
        values = tuple(conditions.values())

    try:
        async with await get_connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, values)
                rows = await cursor.fetchall()
                # Manually map rows to dictionaries
                return [dict(zip(columns, row)) for row in rows]
    except Exception as ex:
        print(f"Error querying table {table}: {ex}")
        return None

@log_function
async def log_non_visitor(face_image, timestamp):
    """
    Logs data for unauthorized attempts to the Non_Visitor_Login_Log table.

    Args:
        face_image (bytes): Captured face image.
        timestamp (datetime): Attempt timestamp.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            sql = "INSERT INTO Non_Visitor_Login_Log (face_image, timestamp) VALUES (?, ?)"
            await cursor.execute(sql, (face_image, timestamp))
            await conn.commit()


@log_function
async def log_to_database(name, face_id):
    """
    Logs visitor data to the database.

    Args:
        name (str): Name of the visitor.
        face_id (int): Associated face ID.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            sql = "INSERT INTO Visitor_Logs (face_id, recognized, timestamp) VALUES (?, ?, ?)"
            timestamp = datetime.datetime.now()
            await cursor.execute(sql, (face_id, name, timestamp))
            await conn.commit()
            print(f"Log entry for {name} added successfully.")

@log_function
async def get_face_id(name):
    """
    Fetches the face ID associated with a given name.

    Args:
        name (str): The name to search for.

    Returns:
        list: The face ID(s) associated with the name.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            sql = "SELECT face_id FROM Faces WHERE name=?"
            await cursor.execute(sql, (name,))
            result = await cursor.fetchall()
            return result

@log_function
async def store_notifications(notification_type, sent_at):
    """
    Stores notification details in the database.

    Args:
        notification_type (str): Type of notification.
        sent_at (datetime.datetime): Timestamp of when the notification was sent.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            sql = "INSERT INTO Notifications (notification_type, sent_at) VALUES (?, ?)"
            await cursor.execute(sql, (notification_type, sent_at))
            await conn.commit()

@log_function
async def update_user_info(user_id, user_data: UserUpdateDTO):
    """
    Updates user information in the database.

    Args:
        user_id (int): ID of the user to update.
        user_data (UserUpdateDTO): Data to update.
    """
    async with await get_connection() as conn:
        async with conn.cursor() as cursor:
            fields = []
            values = []
            
            if user_data.username:
                fields.append("username = ?")
                values.append(user_data.username)
            if user_data.email:
                fields.append("email = ?")
                values.append(user_data.email)
            if user_data.role:
                fields.append("role = ?")
                values.append(user_data.role)
            if user_data.last_login:
                fields.append("last_login = ?")
                values.append(user_data.last_login)

            sql = f"UPDATE Users SET {', '.join(fields)} WHERE user_id = ?"
            values.append(user_id)
            await cursor.execute(sql, tuple(values))
            await conn.commit()
            print("User information updated successfully.")

@log_function
async def login_to_admin(password, user_id, email):
    """
    Authenticates admin credentials.

    Args:
        password (str): Admin password.
        user_id (int): Admin user ID.
        email (str): Admin email.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    sql = 'SELECT role FROM Users WHERE user_id = ? AND password_hash = ? AND email = ?'
    try:
        async with await get_connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, (user_id, password, email))
                result = await cursor.fetchone()
                return True if result and result[0] == 'admin' else False
    except Exception as e:
        return None

@log_function
async def get_userID(username, email, password):
    """
    Fetches the user ID based on username, email, and password.

    Args:
        username (str): Username.
        email (str): Email address.
        password (str): Password.

    Returns:
        int or bool: User ID if found, False otherwise.
    """
    sql = 'SELECT user_id FROM Users WHERE username = ? AND email = ? AND password = ?'
    try:
        async with await get_connection() as conn:
            async with conn.cursor() as cursor:
                await cursor.execute(sql, (username, email, password))
                result = await cursor.fetchone()
                return result if result else False
    except Exception as ex:
        return None
    
@log_function
async def get_notifications_by_user(userID: int):
    """
    Fetches the latest notifications with a given user ID.

    Args:
        userID (int): user ID.

    Returns:
        dict: JSON result containing notifications and associated data.
    """
    # SQL queries
    user_role_query = 'SELECT role FROM Users WHERE user_id = ?'
    notification_data_query = 'SELECT * FROM Notifications WHERE user_id = ?'
    face_image_query = 'SELECT face_image, timestamp FROM Non_Visitor_Login_Log WHERE log_id = ?'
    try:
        async with await get_connection() as conn:
            async with conn.cursor() as cursor:
                # Get user role
                await cursor.execute(user_role_query, (userID,))
                role_result = await cursor.fetchone()
                if not role_result:
                    return {"error": "User not found"}
                role = role_result['role']
                # Only process if the user is an admin
                if role == 'admin':
                    # Fetch notifications for the user
                    await cursor.execute(notification_data_query, (userID,))
                    notifications = await cursor.fetchall()
                    if not notifications:
                        return {"error": "No notifications found for this user"}
                    # Fetch face images and timestamps for the logs in notifications
                    enriched_notifications = []
                    for notification in notifications:
                        log_id = notification['log_id']
                        await cursor.execute(face_image_query, (log_id,))
                        face_data = await cursor.fetchone()
                        if face_data:
                            notification['face_image'] = face_data['face_image']
                            notification['timestamp'] = face_data['timestamp']
                        enriched_notifications.append(notification)
                    return json.dumps({
                        "status": "success",
                        "role": role,
                        "notifications": enriched_notifications
                    }, default=str)
                else:
                    return json.dumps({
                        "status": "error",
                        "message": "User is not authorized to view notifications"
                    })
    except Exception as ex:
        return json.dumps({
            "status": "error",
            "message": str(ex)
        })