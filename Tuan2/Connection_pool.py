import mysql.connector #Sử dụng mô-đun này để giao tiếp với MySQL
from mysql.connector import Error # Mô-đun này sẽ b bất kỳ ngoại lệ cơ sở dữ liệu nào có thể xảy ra trong quá trình này.
from mysql.connector.connection import MySQLConnection
from mysql.connector import pooling #Sử dụng mô-đun này để tạo, quản lý và sử dụng connection pool

try:
    # sử dụng function mysql.connector.pooling.MySQLConnectionPool để tạo một đối tượng connection pool với các tham s
    connection_pool = mysql.connector.pooling.MySQLConnectionPool(pool_name="connection_pool",# teen của connection pool
                                                pool_size=5, # kích thước số kết nối trong connection ponnection pool
                                                pool_reset_session=True,
                                                host='localhost', # ip
                                                database='it4421',# tên csdl
                                                user='root', # tên user
                                                password='') # password truy cập csdl
    print ("Các thuộc tính của connection pool :  ")
    print("Connection Pool Name - ", connection_pool.pool_name)
    print("Connection Pool Size - ", connection_pool.pool_size)

    # lấy về một đối tượng kết nối từ connection_pool đã khởi tạo bằng phương thức get_connection()
    # Phương pháp này trả về một kết nối từ nhóm. Nếu tất cả các kết nối đang được sử dụng hoặc pool trống, nó sẽ tăng PoolError.
    connection_object = connection_pool.get_connection()

    if connection_object.is_connected():
        db_Info = connection_object.get_server_info() # phương thức trả về phiên bản db đang dùng
        print("Kết nối tới MySQL database sử dụng connection pool ... MySQL Server version : ",db_Info)

        cursor = connection_object.cursor() # ****
        cursor.execute("select database();")
        record = cursor.fetchone()
        print ("Đã kết nối tới : ", record)

except Error as e :
    print ("Có lỗi trong khi kết nối tới MySQL sử dụng Connection pool ", e)
finally:
    #closing database connection.
    if(connection_object.is_connected()):
        cursor.close()
        print("MySQL connection đã ngắt kết nối")

