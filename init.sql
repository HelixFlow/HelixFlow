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


-- helix.business_table_asset definition

CREATE TABLE IF NOT EXISTS `business_table_asset` (
  `id` varchar(100) NOT NULL,
  `table_name` varchar(255) NOT NULL,
  `display_name` varchar(255) DEFAULT NULL,
  `dialect` varchar(64) DEFAULT NULL,
  `database_name` varchar(255) DEFAULT NULL,
  `schema_name` varchar(255) DEFAULT NULL,
  `description` text,
  `raw_ddl` text,
  `columns_json` text,
  `primary_keys_json` text,
  `indexes_json` text,
  `tags_json` text,
  `event_time_field` varchar(255) DEFAULT NULL,
  `watermark_expression` varchar(500) DEFAULT NULL,
  `is_dimension` tinyint(1) DEFAULT 0,
  `dimension_key` varchar(255) DEFAULT NULL,
  `ttl_seconds` int DEFAULT NULL,
  `connector_hint` varchar(64) DEFAULT NULL,
  `update_time` varchar(100) DEFAULT NULL,
  `create_time` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_business_table_asset_table_name` (`table_name`),
  KEY `idx_business_table_asset_dialect` (`dialect`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;


-- helix.business_analysis_run definition

CREATE TABLE IF NOT EXISTS `business_analysis_run` (
  `id` varchar(100) NOT NULL,
  `status` varchar(64) DEFAULT NULL,
  `requirement` text,
  `connector_preference` varchar(64) DEFAULT NULL,
  `request_json` text,
  `result_json` text,
  `error` text,
  `update_time` varchar(100) DEFAULT NULL,
  `create_time` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `idx_business_analysis_run_status` (`status`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
