import struct, sys
p = r"H:\Qwen\model\models\Ornith-1.5-9B-Q8_0\Ornith-1.5-9B-Q8_0.gguf"
data = open(p, "rb").read()
magic = data[:4]
assert magic == b"GGUF", magic
ver = struct.unpack("<I", data[4:8])[0]
nt = struct.unpack("<Q", data[8:16])[0]
nkv = struct.unpack("<Q", data[16:24])[0]
print("version", ver, "tensors", nt, "metadata_kv", nkv)
off = 24
def rd_str():
    global off
    n = struct.unpack("<Q", data[off:off+8])[0]; off += 8
    s = data[off:off+n].decode("utf-8", "replace"); off += n
    return s
def rd_val(t):
    global off
    if t==0: v=struct.unpack("<B",data[off:off+1])[0]; off+=1
    elif t==1: v=struct.unpack("<b",data[off:off+1])[0]; off+=1
    elif t==2: v=struct.unpack("<H",data[off:off+2])[0]; off+=2
    elif t==3: v=struct.unpack("<h",data[off:off+2])[0]; off+=2
    elif t==4: v=struct.unpack("<I",data[off:off+4])[0]; off+=4
    elif t==5: v=struct.unpack("<i",data[off:off+4])[0]; off+=4
    elif t==6: v=struct.unpack("<f",data[off:off+4])[0]; off+=4
    elif t==7: v=bool(struct.unpack("<B",data[off:off+1])[0]); off+=1
    elif t==8: v=rd_str()
    elif t==9:
        at=struct.unpack("<I",data[off:off+4])[0]; off+=4
        al=struct.unpack("<Q",data[off:off+8])[0]; off+=8
        v=[rd_val(at) for _ in range(al)]
    elif t==10: v=struct.unpack("<Q",data[off:off+8])[0]; off+=8
    elif t==11: v=struct.unpack("<q",data[off:off+8])[0]; off+=8
    elif t==12: v=struct.unpack("<d",data[off:off+8])[0]; off+=8
    else: v=("?"+str(t)); 
    return v
keys = []
for _ in range(nkv):
    k = rd_str()
    t = struct.unpack("<I", data[off:off+4])[0]; off += 4
    v = rd_val(t)
    keys.append((k, t, v))
for k,t,v in keys:
    if any(s in k for s in ["block_count","head_count","head_count_kv","embedding_length","feed_forward_length","attention","layer","expert","f_norm","context_length","layer_norm"]):
        print(f"  {k} = {v}")
