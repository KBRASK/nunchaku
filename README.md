目前状态: 仅W4A4 linear未对齐(矩阵乘法部分)、W4A16已对齐，ROPE和norm已对齐，attention除to_qkv(属于前面提到的linear)以外已对齐

## 🔥News
- 美东时间 8.25 感谢Muyang、Yujun和Zhekai，将短期优先目标改为将 W4A4 部分先实现成 W16A16
- 美东时间 9.2 感谢Muyang和Yujun, qweigh的unpack非预期数值问题被解决

由于部分未对齐，目前仍需要使用c++ kernel以单独对各部分对齐。
examples/flux.1-dev.py和examples/flux.1-schnell.py
为测试脚本

## Compare
主观对齐后需补充此处客观metric数值