import type { Metadata } from "next";
import { BtcDashboard } from "./BtcDashboard";

export const metadata: Metadata = {
  title: "BTC 회귀채널 복귀 · Paper Desk",
  description: "비트코인 회귀채널 복귀 전략을 관찰하는 페이퍼 트레이딩 대시보드",
};

export default function Home() {
  return <BtcDashboard />;
}
