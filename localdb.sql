CREATE TABLE AUTH_INFO (
	userid char(32) PRIMARY KEY,
   	password char(32) NOT NULL,
	display_name char(32) DEFAULT '',
    role char(16) DEFAULT 'user'
);

INSERT INTO AUTH_INFO values ('charles', '1234!', 'Charles Kim', 'user');
INSERT INTO AUTH_INFO values ('nobody', '1234!', 'Gildong Hong', 'user');


CREATE TABLE USER_REQUEST (
	userid char(32) NOT NULL,
	request_time DATETIME default current_timestamp,
	request_type char(64),
	registration_nm char(16),
	status char(16) NOT NULL default '접수',
	FOREIGN KEY (userid)
		REFERENCES AUTH_INFO (userid) 
			ON DELETE CASCADE 
         	ON UPDATE NO ACTION
);

INSERT INTO USER_REQUEST values( 'charles', datetime('now','localtime'), '거래내역서', '6864005911801', '처리중');
INSERT INTO USER_REQUEST values( 'charles', datetime('now', 'localtime', '-5 days'), '거래내역서', '6864005911802', '처리완료');
INSERT INTO USER_REQUEST values( 'charles', datetime('now', 'localtime', '-24 days'), '거래내역서', '6864005911802', '처리완료');
