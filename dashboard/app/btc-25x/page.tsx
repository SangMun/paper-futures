import type { Metadata } from "next";
import { Btc25xDashboard } from "./Btc25xDashboard";

export const metadata: Metadata = {
  title: "BTC 25× 국면 채널 · Paper Desk",
  description: "BTC 25배 단기 국면·회귀채널 전략 관찰 대시보드",
};

export default function Btc25xPage() {
  return <Btc25xDashboard />;
}
