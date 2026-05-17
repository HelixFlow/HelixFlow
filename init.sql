CREATE DATABASE IF NOT EXISTS `helix`;
USE `helix`;
-- helix.flow definition

CREATE TABLE IF NOT EXISTS `flow` (
  `id` varchar(100) NOT NULL,
  `data` text,
  `name` varchar(100) DEFAULT NULL,
  `user_id` int DEFAULT NULL,
  `description` varchar(100) DEFAULT NULL,
  `logo` varchar(500) DEFAULT NULL,
  `status` int DEFAULT NULL,
  `update_time` varchar(100) DEFAULT NULL,
  `create_time` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


-- helix.`user` definition

CREATE TABLE IF NOT EXISTS `user` (
  `id` int NOT NULL,
  `name` varchar(100) DEFAULT NULL,
  `password` varchar(100) DEFAULT NULL,
  `nick_name` varchar(100) DEFAULT NULL,
  `phone` varchar(100) DEFAULT NULL,
  `email` varchar(100) DEFAULT NULL,
  `status` int DEFAULT NULL,
  `remark` varchar(100) DEFAULT NULL,
  `expire_time` varchar(100) DEFAULT NULL,
  `create_time` varchar(100) DEFAULT NULL,
  `update_time` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
