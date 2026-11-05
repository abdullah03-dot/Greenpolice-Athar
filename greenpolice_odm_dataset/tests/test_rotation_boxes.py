"""
Reproduces the rotation table at the top of 03_augmentation_pipeline.py: how far the axis-aligned
box that Albumentations re-fits after a free rotation is from the TRUE tight box of the rotated
object ('largest_box' = rotate the 4 box corners, 'ellipse' = the inscribed-ellipse method).
Mean IoU well below 0.95 = label noise for mAP@0.5:0.95.   Run: python tests/test_rotation_boxes.py
"""
import numpy as np
rng=np.random.default_rng(0)
def shape(kind):
    t=np.linspace(0,2*np.pi,72,endpoint=False)
    if kind=="rect":   # boxes/monitors/laptops (render with yaw already)
          w,h=1,rng.uniform(0.3,1); a=rng.uniform(0,np.pi)
          P=np.array([[-w,-h],[w,-h],[w,h],[-w,h]])/2
    elif kind=="thin":  # cables, tubes, bulbs-in-line
          w,h=1,rng.uniform(0.02,0.08); a=rng.uniform(0,np.pi)
          P=np.array([[-w,-h],[w,-h],[w,h],[-w,h]])/2
    elif kind=="blob":  # piles, bundles
          r=1+0.3*np.sin(3*t+rng.uniform(0,6))*rng.uniform(0,1); a=0
          P=np.c_[r*np.cos(t),r*np.sin(t)*rng.uniform(.5,1)]
    R=lambda q:np.array([[np.cos(q),-np.sin(q)],[np.sin(q),np.cos(q)]])
    return P@R(a).T
def box(P): return np.r_[P.min(0),P.max(0)]
def iou(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0]));iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    I=ix*iy; return I/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-I)
def main():
  for lim in (10,45):
    for kind in ("rect","thin","blob"):
      L,E=[],[]
      for _ in range(3000):
          P=shape(kind); B=box(P); c=(B[:2]+B[2:])/2; w,h=B[2:]-B[:2]
          q=np.deg2rad(rng.uniform(-lim,lim)); R=np.array([[np.cos(q),-np.sin(q)],[np.sin(q),np.cos(q)]])
          true=box(P@R.T)
          C=np.array([[B[0],B[1]],[B[2],B[1]],[B[2],B[3]],[B[0],B[3]]])@R.T; lb=box(C)
          a,b=w/2,h/2; hw=np.sqrt((a*np.cos(q))**2+(b*np.sin(q))**2); hh=np.sqrt((a*np.sin(q))**2+(b*np.cos(q))**2)
          cc=c@R.T; el=np.r_[cc-[hw,hh],cc+[hw,hh]]
          L.append(iou(lb,true));E.append(iou(el,true))
      L,E=np.array(L),np.array(E)
      print(f"±{lim:>2}° {kind:5s} largest_box IoU mean {L.mean():.3f} p5 {np.percentile(L,5):.3f} | ellipse mean {E.mean():.3f} p5 {np.percentile(E,5):.3f}")


if __name__ == "__main__":
    main()
