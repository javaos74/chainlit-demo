CREATE TABLE authdb (
	userid char(32) PRIMARY KEY,
   	password char(32) NOT NULL,
	display_name char(32) DEFAULT '',
    role char(16) DEFAULT 'user'
);

INSERT INTO authdb values ('charles', '1234!', 'Charles Kim', 'user');
INSERT INTO authdb values ('nobody', '1234!', 'Gildong Hong', 'user');