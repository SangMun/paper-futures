import type { Metadata } from "next";
import { XrpPatternDashboard } from "./XrpPatternDashboard";

export const metadata: Metadata = {
  title: "XRP 패턴 스캐너 · Paper Desk",
  description: "XRP 다중 시간봉 패턴 단타 전략 관찰 대시보드",
};

export default function XrpPatternPage() {
  return <XrpPatternDashboard />;
}
