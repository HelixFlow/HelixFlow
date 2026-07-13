import unittest
from unittest.mock import patch

from service.business_analysis.ddl_parser import parse_business_ddl
from service.business_analysis.generator import build_analysis_result


MYSQL_DDL = """
CREATE TABLE `order_info` (
  `order_id` bigint NOT NULL COMMENT '订单ID',
  `user_id` bigint DEFAULT NULL COMMENT '用户ID',
  `pay_amount` decimal(18,2) DEFAULT NULL COMMENT '支付金额',
  `event_time` datetime NOT NULL COMMENT '事件时间',
  PRIMARY KEY (`order_id`),
  KEY `idx_user` (`user_id`)
) COMMENT='订单表';
"""


ORACLE_DDL = """
CREATE TABLE DW.ORDER_PAY (
  ORDER_ID NUMBER(20) NOT NULL,
  PAY_TIME TIMESTAMP,
  AMOUNT NUMBER(18, 2),
  USER_NAME VARCHAR2(64),
  CONSTRAINT PK_ORDER_PAY PRIMARY KEY (ORDER_ID)
);
COMMENT ON TABLE DW.ORDER_PAY IS '支付流水表';
COMMENT ON COLUMN DW.ORDER_PAY.PAY_TIME IS '支付时间';
"""


DAMENG_DDL = """
CREATE TABLE DM_USER (
  ID BIGINT NOT NULL,
  USER_NAME VARCHAR(64),
  CREATE_TIME DATETIME,
  PRIMARY KEY(ID)
);
"""


TDSQL_DDL = """
CREATE TABLE account_balance (
  acct_id varchar(64) NOT NULL COMMENT '账户',
  balance decimal(20, 4) COMMENT '余额',
  update_time timestamp COMMENT '更新时间',
  PRIMARY KEY (acct_id)
);
"""


class BusinessAnalysisTest(unittest.TestCase):
    def test_parse_mysql_ddl(self):
        parsed = parse_business_ddl(MYSQL_DDL, "mysql")
        self.assertEqual(parsed["table_name"], "order_info")
        self.assertEqual(parsed["primary_keys"], ["order_id"])
        self.assertEqual(parsed["event_time_field"], "event_time")
        self.assertEqual(parsed["columns"][2]["flink_type"], "DECIMAL(18,2)")
        self.assertEqual(parsed["indexes"][0]["columns"], ["user_id"])

    def test_parse_oracle_ddl_with_comment_on_column(self):
        parsed = parse_business_ddl(ORACLE_DDL, "oracle")
        self.assertEqual(parsed["schema_name"], "DW")
        self.assertEqual(parsed["table_name"], "ORDER_PAY")
        self.assertEqual(parsed["description"], "支付流水表")
        pay_time = next(col for col in parsed["columns"] if col["name"] == "PAY_TIME")
        self.assertEqual(pay_time["comment"], "支付时间")
        self.assertEqual(pay_time["flink_type"], "TIMESTAMP(3)")

    def test_parse_dameng_and_tdsql(self):
        dameng = parse_business_ddl(DAMENG_DDL, "dameng")
        tdsql = parse_business_ddl(TDSQL_DDL, "tdsql")
        self.assertEqual(dameng["event_time_field"], "CREATE_TIME")
        self.assertEqual(tdsql["primary_keys"], ["acct_id"])
        self.assertEqual(tdsql["columns"][1]["flink_type"], "DECIMAL(20,4)")

    def test_generation_defaults_to_kafka(self):
        parsed = parse_business_ddl(MYSQL_DDL, "mysql")
        result = build_analysis_result("统计每分钟订单支付金额", [parsed], connector_preference="auto")
        self.assertEqual(result["selected_connector_templates"][0]["connector_type"], "kafka")
        self.assertIn("'connector' = 'kafka'", "\n".join(result["flink_create_tables"]))
        self.assertIn("INSERT INTO order_info_result", result["flink_insert_sql"])

    def test_generation_uses_upsert_for_update_semantics(self):
        parsed = parse_business_ddl(TDSQL_DDL, "tdsql")
        result = build_analysis_result("根据账户主键更新余额，生成聚合结果更新流", [parsed], connector_preference="auto")
        self.assertEqual(result["selected_connector_templates"][0]["connector_type"], "upsert-kafka")
        self.assertIn("'connector' = 'upsert-kafka'", "\n".join(result["flink_create_tables"]))
        self.assertIn("PRIMARY KEY", "\n".join(result["flink_create_tables"]))

    def test_agent_rag_falls_back_without_key(self):
        parsed = parse_business_ddl(MYSQL_DDL, "mysql")
        result = build_analysis_result(
            "统计每分钟订单支付金额",
            [parsed],
            connector_preference="auto",
            use_agent_rag=True,
        )
        self.assertEqual(result["generation_mode"], "rule_draft")
        self.assertFalse(result["agent_trace"]["used"])
        self.assertIn("缺少模型 API Key", result["risks"][-1]["message"])

    def test_selected_agent_flow_can_drive_generation(self):
        parsed = parse_business_ddl(MYSQL_DDL, "mysql")
        agent_flow_data = {
            "nodes": [
                {"id": "start", "data": {"name": "start", "display_name": "start", "input": [], "output": None, "params": None}},
                {"id": "call_model_1", "data": {"name": "call_model", "display_name": "call_model_1", "input": [], "output": [], "params": []}},
            ],
            "edges": [],
        }
        flow_output = {
            "input": """{
              "requirement_summary": "agent generated",
              "flink_create_tables": ["CREATE TABLE agent_src (id BIGINT) WITH ('connector' = 'kafka');"],
              "flink_insert_sql": "INSERT INTO sink SELECT * FROM agent_src;",
              "dimension_table_plan": [],
              "ttl_plan": [],
              "resource_plan": {"parallelism": 1},
              "risks": [],
              "assumptions": ["from agent flow"]
            }"""
        }

        with patch("service.business_analysis.agent_rag.flow_run_manager.run_process_compat", return_value=flow_output) as mocked:
            result = build_analysis_result(
                "统计每分钟订单支付金额",
                [parsed],
                connector_preference="auto",
                use_agent_rag=True,
                openai_api_key="sk-test",
                agent_flow_id="flow-1",
                agent_flow_data=agent_flow_data,
            )

        self.assertEqual(result["generation_mode"], "agent_flow")
        self.assertEqual(result["requirement_summary"], "agent generated")
        self.assertEqual(result["agent_trace"]["selected_flow_id"], "flow-1")
        mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
